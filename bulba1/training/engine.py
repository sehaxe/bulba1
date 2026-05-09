import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.amp import autocast
from typing import Optional
from tqdm import tqdm
from pathlib import Path

from bulba1.training.optimizer import CombinedOptimizer
from bulba1.training.ema import EMA
from bulba1.training.checkpoint import CheckpointManager
from bulba1.training.stages import stage_for_step, compute_curriculum_seq_len
from bulba1.training.chunked_ce import chunked_cross_entropy
from bulba1.training.eval import run_eval
from bulba1.training.monitor import SystemMonitor, preflight_memory_test


class TrainingEngine:
    def __init__(self, model, cfg, tokenizer, device="cuda"):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.step = 0
        self.best_loss = float("inf")
        self.ema_loss = None

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        # Параметры обучения исключительно из cfg
        gs = getattr(cfg, "grad_accum_steps", 1)
        self.grad_accum_steps = max(1, gs)

        self.total_vram = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024 if self.device.type == "cuda" else 0
        self.optimizer = CombinedOptimizer(model, cfg)
        self.ema = EMA(model, decay=cfg.ema_decay)
        self.checkpoint_mgr = CheckpointManager(cfg.checkpoint_dir, keep_top_k=cfg.checkpoint_keep_top_k)

        self.use_amp = cfg.use_f16
        self.use_chunked_ce = cfg.vocab_size > cfg.auto_chunked_ce_threshold if hasattr(cfg, "auto_chunked_ce_threshold") else False
        self.chunk_size = getattr(cfg, "chunk_size", 8192)

        self.monitor = SystemMonitor(self.device, interval_sec=5.0)

        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Приводим веса всех норм к bfloat16 (убирает предупреждение)
        for module in self.model.modules():
            if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)

        self.start_time = time.time()
        self.tokens_processed = 0
        self.oom_count = 0
        self.batch_reductions = 0

    # ------------------------------------------------------------------
    def _optimizer_step(self, step=0, total_steps=1):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm
        )
        noise_level = getattr(self.cfg, 'gradient_noise', 0.0)
        if noise_level > 0.0:
            decay = 1.0 - step / max(1, total_steps)
            std = noise_level * decay
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * std)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return grad_norm

    def update_bitnet_activation_bits(self, step: int, total: int):
        if not getattr(self.cfg, "use_bitnet_a48", False) or not getattr(self.cfg, "a48_two_stage_training", False):
            return
        stage1_ratio = getattr(self.cfg, "a48_stage1_steps_ratio", 0.95)
        stage1_bits = getattr(self.cfg, "a48_stage1_bits", 8)
        stage2_bits = getattr(self.cfg, "a48_stage2_bits", 4)
        target_bits = stage1_bits if step < int(total * stage1_ratio) else stage2_bits
        from bulba1.model.bit_linear import BitLinear
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                module.activation_bits = target_bits

    # ------------------------------------------------------------------
    def _reduce_batch_size(self) -> bool:
        if self.batch_reductions >= self.cfg.max_batch_reductions:
            return False
        old_bs = self.cfg.batch_size
        new_bs = max(self.cfg.min_batch_size, old_bs // 2)
        if new_bs == old_bs:
            return False
        self.cfg.batch_size = new_bs
        self.grad_accum_steps = max(1, self.grad_accum_steps * 2)
        self.batch_reductions += 1
        print(f"[OOM] batch_size {old_bs} -> {new_bs}, accum_steps -> {self.grad_accum_steps}")
        torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------
    def safe_forward(self, input_ids: torch.Tensor):
        if self.use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                return self.model(input_ids, self.cfg.checkpoint_every_n_layers)
        return self.model(input_ids, self.cfg.checkpoint_every_n_layers)

    # ------------------------------------------------------------------
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_chunked_ce:
            return chunked_cross_entropy(
                logits, targets,
                chunk_size=self.chunk_size,
                ignore_index=self.cfg.ignore_index,
            )
        return F.cross_entropy(
            logits, targets, label_smoothing=self.cfg.label_smoothing
        )

    # ------------------------------------------------------------------
    def compute_curriculum_seq_len(self, step: int, total: int) -> int:
        seq_lens = getattr(self.cfg, "curriculum_seq_lens", None)
        boundaries = getattr(self.cfg, "curriculum_seq_len_boundaries", None)
        if seq_lens and len(seq_lens) > 0:
            return compute_curriculum_seq_len(
                step, total,
                seq_lens=seq_lens,
                boundaries=boundaries,
                target_seq_len=self.cfg.seq_len,
            )
        if self.cfg.curriculum_warmup_ratio <= 0:
            return self.cfg.seq_len
        warmup = max(1, int(total * self.cfg.curriculum_warmup_ratio))
        if step >= warmup:
            return self.cfg.seq_len
        progress = (step / warmup) ** 3
        return max(
            self.cfg.curriculum_start_seq_len,
            int(
                self.cfg.curriculum_start_seq_len
                + (self.cfg.seq_len - self.cfg.curriculum_start_seq_len) * progress
            ),
        )

    # ------------------------------------------------------------------
    def compute_lr(self, step: int, total: int) -> float:
        base_lr = self.cfg.learning_rate          # больше не умножаем на stage-множитель
        warmup = int(total * self.cfg.warmup_ratio) if self.cfg.warmup_ratio > 0 else 0
        if warmup > 0 and step < warmup:
            return base_lr * (step / warmup)
        # cosine decay до 1% от base_lr
        progress = (step - warmup) / max(1, total - warmup)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * max(decay, 0.01)

    # ------------------------------------------------------------------
    def train_step(self, batch, is_accum_last: bool, step: int, total_steps: int):
        self.model.train()

        input_ids, targets = batch if isinstance(batch, (list, tuple)) else (batch, batch)

        if input_ids.size(0) > self.cfg.batch_size:
            input_ids = input_ids[: self.cfg.batch_size]
            targets = targets[: self.cfg.batch_size]

        curr_seq_len = self.compute_curriculum_seq_len(step, total_steps)
        if input_ids.size(1) > curr_seq_len:
            input_ids = input_ids[:, :curr_seq_len]
            targets = targets[:, :curr_seq_len]

        B, T = input_ids.shape
        clr = self.cfg.num_clr_tokens

        _, _, vram_pct = self.get_vram_usage()
        if vram_pct > self.cfg.vram_critical_pct:
            self.oom_count += 1
            return {"loss": float("inf"), "oom": True}

        try:
            logits, mtp1, mtp2, aux_loss = self.safe_forward(input_ids)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.oom_count += 1
            if self._reduce_batch_size():
                return {"loss": float("inf"), "oom": True, "reduced_batch": True}
            return {"loss": float("inf"), "oom": True}

        text_logits = logits[:, clr : clr + T, :].reshape(-1, self.cfg.vocab_size)
        loss_main = self.compute_loss(text_logits, targets.reshape(-1))
        loss_total = loss_main

        if self.cfg.use_skip_gram and self.cfg.skip_gram_range > 1:
            for k in range(2, self.cfg.skip_gram_range + 1):
                sg_loss = self._skip_gram_loss(
                    logits[:, clr : clr + T, :], targets, k - 1
                )
                if sg_loss.item() > 0:
                    loss_total = loss_total + self.cfg.skip_gram_weight * sg_loss

        if mtp1 is not None:
            mtp1_text = mtp1[:, clr : clr + T, :].reshape(-1, self.cfg.vocab_size)
            loss_mtp1 = self.compute_loss(mtp1_text, targets.reshape(-1))
            mtp_scale = min(1.0, step / max(1, self.cfg.mtp1_warmup_steps))
            loss_total = loss_total + mtp_scale * self.cfg.loss_mtp1_weight * loss_mtp1

        if mtp2 is not None:
            mtp2_text = mtp2[:, clr : clr + T, :].reshape(-1, self.cfg.vocab_size)
            loss_mtp2 = self.compute_loss(mtp2_text, targets.reshape(-1))
            mtp2_scale = min(1.0, step / max(1, self.cfg.mtp2_warmup_steps))
            loss_total = loss_total + mtp2_scale * self.cfg.loss_mtp2_weight * loss_mtp2

        loss_total = loss_total + aux_loss * self.cfg.router_z_loss_coef

        try:
            scaled_loss = loss_total / self.grad_accum_steps
            scaled_loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.oom_count += 1
            if self._reduce_batch_size():
                return {"loss": float("inf"), "oom": True, "reduced_batch": True}
            return {"loss": float("inf"), "oom": True}

        if is_accum_last:
            try:
                grad_norm = self._optimizer_step(step, total_steps)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.oom_count += 1
                if self._reduce_batch_size():
                    return {"loss": float("inf"), "oom": True, "reduced_batch": True}
                return {"loss": float("inf"), "oom": True}

            if self.ema:
                self.ema.update(self.model)

            self.tokens_processed += B * T
            return {"loss": loss_main.item(), "grad_norm": grad_norm, "oom": False}

        return {"loss": loss_main.item(), "oom": False}

    # ------------------------------------------------------------------
    def _skip_gram_loss(self, logits, targets, k):
        if k <= 0 or logits.size(1) <= k:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        pred = logits[:, :-k, :].reshape(-1, self.cfg.vocab_size)
        tgt = targets[:, k:].reshape(-1)
        return F.cross_entropy(pred, tgt)

    # ------------------------------------------------------------------
    def get_vram_usage(self) -> tuple:
        if self.device.type != "cuda":
            return 0, 0, 0.0
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        pct = allocated / self.total_vram * 100 if self.total_vram > 0 else 0
        return allocated, reserved, pct

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader,
        eval_loader=None,
        eval_prompts=None,
        resume_step: int = 0,
    ):
        total_steps = self.cfg.total_steps
        eval_every = getattr(self.cfg, "eval_every", 0)
        checkpoint_every = self.cfg.checkpoint_every
        log_every = getattr(self.cfg, "log_every", 100)
        early_log_steps = getattr(self.cfg, "early_log_steps", 0)
        early_log_every = getattr(self.cfg, "early_log_every", log_every)

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(f"Training started on {gpu_name}: "
              f"{self.cfg.format_params()} model, "
              f"batch={self.cfg.batch_size}, seq_len={self.cfg.seq_len}, "
              f"total_steps={total_steps}")
        print(f"Logs: {self.log_dir / 'bulba1.jsonl'}")

        if not getattr(self.cfg, "skip_preflight", False):
            print("[PRE-FLIGHT] Running memory test...")
            preflight = preflight_memory_test(self.model, self.cfg, self.device, max_attempts=3)
            if not preflight["success"]:
                print(f"[ERROR] Pre-flight FAILED: {preflight['error']}")
                return self.model
            print("[PRE-FLIGHT] PASSED")
        else:
            print(f"[SKIP] Pre-flight disabled. Using batch_size={self.cfg.batch_size}")

        self.step_times = []
        self.step_start_time = time.time()

        pbar = tqdm(total=total_steps, desc="Training", unit="steps", initial=resume_step,
                    dynamic_ncols=True)

        for step in range(resume_step, total_steps):
            self.step = step
            accum_loss = 0.0
            valid_steps = 0
            stage = stage_for_step(step, total_steps)

            for accum_step in range(self.grad_accum_steps):
                try:
                    batch = next(train_loader)
                except StopIteration:
                    break

                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
                else:
                    batch = batch.to(self.device)

                lr = self.compute_lr(step, total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                self.update_bitnet_activation_bits(step, total_steps)

                is_last = accum_step == self.grad_accum_steps - 1
                result = self.train_step(
                    batch, is_accum_last=is_last, step=step, total_steps=total_steps
                )

                if result.get("reduced_batch"):
                    self.optimizer.zero_grad()
                    valid_steps = 0
                    break

                if result["oom"]:
                    continue

                accum_loss += result["loss"]
                valid_steps += 1

            if valid_steps == 0:
                continue

            loss_val = accum_loss / valid_steps
            if loss_val < self.best_loss:
                self.best_loss = loss_val
            self.ema_loss = (
                0.9 * (self.ema_loss or loss_val) + 0.1 * loss_val
            )

            if (step + 1) % checkpoint_every == 0:
                self.checkpoint_mgr.save(
                    self.model, self.optimizer, step + 1, loss_val,
                    config={"lr": lr, "stage": stage.name() if hasattr(stage, "name") else str(stage)},
                    ema=self.ema,
                )
                print(f"[CHECKPOINT] Saved at step {step+1}")

            if eval_every > 0 and (step + 1) % eval_every == 0 and eval_loader:
                self._run_eval(eval_loader, eval_prompts)

            current_log_every = early_log_every if early_log_steps > 0 and step < early_log_steps else log_every
            if (step + 1) % current_log_every == 0:
                self._log_status(step, total_steps, loss_val, stage, lr)

            pbar.update(1)
            elapsed = time.time() - self.start_time
            tok_per_sec = self.tokens_processed / elapsed if elapsed else 0
            pbar.set_postfix(loss=f"{loss_val:.4f}", ema=f"{self.ema_loss:.4f}",
                             lr=f"{lr:.2e}", tok_s=f"{tok_per_sec:.0f}")

        pbar.close()
        print("[DONE] Training complete!")
        if self.ema:
            self.ema.apply_shadow(self.model)
        return self.model

    # ------------------------------------------------------------------
    def _log_status(self, step, total_steps, loss_val, stage, lr):
        elapsed = time.time() - self.start_time
        tok_per_sec = self.tokens_processed / elapsed if elapsed else 0
        snap = self.monitor.snapshot(force=True)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step + 1,
            "total_steps": total_steps,
            "loss": round(loss_val, 6),
            "ema_loss": round(self.ema_loss, 6) if self.ema_loss else None,
            "best_loss": round(self.best_loss, 6),
            "lr": lr,
            "stage": stage.name(),
            "optimizer": "Muon+AdamW",
            "vram_used_mb": int(snap.vram_reserved_mb),
            "vram_total_mb": int(snap.vram_total_mb),
            "vram_pct": round(snap.vram_pct, 1),
            "ram_used_mb": int(snap.ram_used_mb),
            "ram_total_mb": int(snap.ram_total_mb),
            "ram_pct": round(snap.ram_pct, 1),
            "cpu_pct": int(snap.cpu_pct),
            "gpu_util_pct": int(snap.gpu_util_pct) if snap.gpu_util_pct is not None else None,
            "tok_per_sec": int(tok_per_sec),
            "oom_count": self.oom_count,
            "batch_size": self.cfg.batch_size,
        }
        print(
            f"[{record['timestamp'][-8:]}] Step {record['step']}/{total_steps}"
            f" | loss={record['loss']:.4f} | ema={record['ema_loss']:.4f}"
            f" | best={record['best_loss']:.4f} | LR={lr:.2e}"
            f" | {record['stage']} | tok/s={record['tok_per_sec']}"
            f" | VRAM={record['vram_used_mb']}MB | OOM={self.oom_count}",
            flush=True
        )
        log_file = self.log_dir / "bulba1.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()