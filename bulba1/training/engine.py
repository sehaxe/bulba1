import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast

from bulba1.autonomy import AutoPilot
from bulba1.training.checkpoint import CheckpointManager
from bulba1.training.chunked_ce import chunked_cross_entropy
from bulba1.training.ema import EMA
from bulba1.training.eval import compute_perplexity, generate_samples, run_eval
from bulba1.training.monitor import SystemMonitor, preflight_memory_test
from bulba1.training.optimizer import CombinedOptimizer
from bulba1.training.stages import stage_for_step


class TrainingEngine:
    def __init__(self, model, cfg, tokenizer, device="cuda", auto_mode: bool = False):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.step = 0
        self.best_loss = float("inf")
        self.ema_loss = None
        self.auto_mode = auto_mode

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        gs = getattr(cfg, "grad_accum_steps", 1)
        self.grad_accum_steps = max(1, gs)

        self.total_vram = (
            torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
            if self.device.type == "cuda"
            else 0
        )
        self.optimizer = CombinedOptimizer(model, cfg)
        self.ema = EMA(model, decay=cfg.ema_decay)
        self.checkpoint_mgr = CheckpointManager(
            cfg.checkpoint_dir, keep_top_k=cfg.checkpoint_keep_top_k
        )

        self.use_amp = cfg.use_f16
        self.use_chunked_ce = (
            cfg.vocab_size > cfg.auto_chunked_ce_threshold
            if hasattr(cfg, "auto_chunked_ce_threshold")
            else False
        )
        self.chunk_size = getattr(cfg, "chunk_size", 8192)

        self.monitor = SystemMonitor(self.device, interval_sec=5.0)

        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_eval = run_eval

        for module in self.model.modules():
            if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)

        self.start_time = time.time()
        self.tokens_processed = 0
        self.oom_count = 0
        self.batch_reductions = 0
        self.model_params = sum(p.numel() for p in model.parameters())
        self.chinchilla_target = self.model_params * 20 * getattr(cfg, "epochs", 2)

    def _update_grad_accum(self, step: int, total: int):
        if not getattr(self.cfg, "dynamic_batch", True):
            return
        progress = step / max(1, total)
        base_accum = max(2, getattr(self.cfg, "base_grad_accum", 4))
        if progress < 0.3:
            self.grad_accum_steps = base_accum
        elif progress < 0.7:
            self.grad_accum_steps = max(2, base_accum - 1)
        else:
            self.grad_accum_steps = max(1, base_accum - 2)

        if getattr(self.cfg, "use_chinchilla_steps", True):
            self._recalc_chinchilla_steps()
            print(f"[Chinchilla] {self.model_params/1e6:.1f}M params × 20 = {self.chinchilla_target/1e9:.2f}B tokens")
            print(f"[Chinchilla] {self.cfg.batch_size}×{self.grad_accum_steps}×{self.cfg.seq_len} = {self.cfg.batch_size * self.grad_accum_steps * self.cfg.seq_len} tok/step → {self.cfg.total_steps} steps")

        if self.auto_mode:
            self.autopilot = AutoPilot(self.cfg, log_dir=self.cfg.log_dir)
        else:
            self.autopilot = None

    # ------------------------------------------------------------------
    def _optimizer_step(self, step=0, total_steps=1):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        noise_level = float(getattr(self.cfg, "gradient_noise", 0.0))
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
        if not getattr(self.cfg, "use_bitnet_a48", False) or not getattr(
            self.cfg, "a48_two_stage_training", False
        ):
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
    def _recalc_chinchilla_steps(self):
        tokens_per_step = self.cfg.batch_size * self.grad_accum_steps * self.cfg.seq_len
        remaining = max(0, self.chinchilla_target - self.tokens_processed)
        self.cfg.total_steps = self.step + math.ceil(remaining / tokens_per_step)

    # ------------------------------------------------------------------
    def _reduce_batch_size(self) -> bool:
        if self.batch_reductions >= self.cfg.max_batch_reductions:
            return False
        old_bs = self.cfg.batch_size
        new_bs = max(self.cfg.min_batch_size, old_bs - 2)
        if new_bs == old_bs:
            return False
        self.cfg.batch_size = new_bs
        self.batch_reductions += 1
        if getattr(self.cfg, "use_chinchilla_steps", True):
            self._recalc_chinchilla_steps()
        print(f"[OOM] batch_size {old_bs} -> {new_bs}, total_steps -> {self.cfg.total_steps}")
        torch.cuda.empty_cache()
        return True

    def _proactive_batch_check(self) -> bool:
        if self.device.type != "cuda":
            return False
        allocated, _, pct = self.get_vram_usage()
        if pct > 82:
            torch.cuda.empty_cache()
        peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        peak_pct = peak_mb / self.total_vram * 100
        torch.cuda.reset_peak_memory_stats(self.device)
        if peak_pct > self.cfg.vram_critical_pct:
            print(f"[VRAM] Peak {peak_pct:.1f}% > critical {self.cfg.vram_critical_pct}%, reducing batch")
            return self._reduce_batch_size()
        return False

    # ------------------------------------------------------------------
    def safe_forward(self, input_ids: torch.Tensor):
        if self.use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                return self.model(input_ids)
        return self.model(input_ids)

    # ------------------------------------------------------------------
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_chunked_ce:
            return chunked_cross_entropy(
                logits,
                targets,
                chunk_size=self.chunk_size,
                ignore_index=self.cfg.ignore_index,
            )
        return F.cross_entropy(logits, targets, label_smoothing=self.cfg.label_smoothing)

    def _forward_and_loss(self, input_ids, targets, T, clr, step, total_steps, curr_seq_len):
        tok_drop = getattr(self.cfg, "token_dropout", 0.0)
        if self.model.training and tok_drop > 0:
            mask = torch.rand_like(input_ids, dtype=torch.float32) > tok_drop
            input_ids = torch.where(mask, input_ids, torch.full_like(input_ids, self.cfg.pad_id))
        logits, mtp1, mtp2, aux_loss = self.safe_forward(input_ids)
        text_logits = logits[:, clr : clr + T, :].reshape(-1, self.cfg.vocab_size)
        loss_main = self.compute_loss(text_logits, targets.reshape(-1))
        loss_total = loss_main
        if self.cfg.use_skip_gram and self.cfg.skip_gram_range > 1 and curr_seq_len >= 128:
            for k in range(2, self.cfg.skip_gram_range + 1):
                sg_loss = self._skip_gram_loss(logits[:, clr : clr + T, :], targets, k - 1)
                if sg_loss.detach() > 0:
                    loss_total = loss_total + self.cfg.skip_gram_weight * sg_loss
        if mtp1 is not None:
            loss_mtp1 = self.compute_loss(mtp1[:, clr : clr + T, :].reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
            scale = self._mtp_scale(step, total_steps, self.cfg.mtp1_warmup_steps)
            loss_total = loss_total + scale * self.cfg.loss_mtp1_weight * loss_mtp1
        if mtp2 is not None:
            loss_mtp2 = self.compute_loss(mtp2[:, clr : clr + T, :].reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
            scale = self._mtp_scale(step, total_steps, self.cfg.mtp2_warmup_steps)
            loss_total = loss_total + scale * self.cfg.loss_mtp2_weight * loss_mtp2
        loss_total = loss_total + aux_loss * self.cfg.router_z_loss_coef
        return loss_main, loss_total

    def compute_curriculum_seq_len(self, step: int, total: int) -> int:
        ratio = getattr(self.cfg, "curriculum_warmup_ratio", 0.02)
        if ratio <= 0:
            return self.cfg.seq_len
        warmup = max(1, int(total * ratio))
        if step >= warmup:
            return self.cfg.seq_len
        progress = (step / warmup)
        return max(
            self.cfg.curriculum_start_seq_len,
            int(self.cfg.curriculum_start_seq_len
                + (self.cfg.seq_len - self.cfg.curriculum_start_seq_len) * (progress ** 0.5)),
        )

    # ------------------------------------------------------------------
    def compute_lr(self, step: int, total: int) -> float:
        if self.autopilot is not None:
            return self.autopilot.compute_lr(step)

        base_lr = self.cfg.learning_rate
        warmup = int(total * self.cfg.warmup_ratio) if self.cfg.warmup_ratio > 0 else 0
        if warmup > 0 and step < warmup:
            return base_lr * (step / warmup)

        use_cooldown = getattr(self.cfg, "use_lr_cooldown", False)
        cooldown_ratio = getattr(self.cfg, "lr_cooldown_ratio", 0.05)
        cooldown_min = getattr(self.cfg, "lr_cooldown_min", 0.01)
        if use_cooldown and step >= total * (1 - cooldown_ratio):
            return base_lr * cooldown_min

        progress = (step - warmup) / max(1, total - warmup)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * max(decay, 0.01)

    # ------------------------------------------------------------------
    def _mtp_scale(self, step: int, total: int, warmup_steps: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        use_cooldown = getattr(self.cfg, "use_mtp_cooldown", False)
        if not use_cooldown:
            return 1.0
        cooldown_ratio = getattr(self.cfg, "mtp_cooldown_ratio", 0.1)
        end_scale = getattr(self.cfg, "mtp_end_scale", 0.1)
        if step >= total * (1 - cooldown_ratio):
            cooldown_steps = total * cooldown_ratio
            progress = (step - (total - cooldown_steps)) / max(1, cooldown_steps)
            scale = 1.0 - progress * (1.0 - end_scale)
            return max(end_scale, scale)
        return 1.0

    # ------------------------------------------------------------------
    def train_step(
        self,
        batch,
        is_accum_last: bool,
        step: int,
        total_steps: int,
        curr_seq_len: int | None = None,
    ):
        self.model.train()

        input_ids, targets = batch if isinstance(batch, (list, tuple)) else (batch, batch)

        if input_ids.size(0) > self.cfg.batch_size:
            input_ids = input_ids[: self.cfg.batch_size]
            targets = targets[: self.cfg.batch_size]

        if curr_seq_len is None:
            curr_seq_len = self.compute_curriculum_seq_len(step, total_steps)

        if input_ids.size(1) > curr_seq_len:
            input_ids = input_ids[:, :curr_seq_len]
            targets = targets[:, :curr_seq_len]

        B, T = input_ids.shape
        clr = self.cfg.num_clr_tokens

        if self.step % 10 == 0:
            _, _, vram_pct = self.get_vram_usage()
            if self.device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
                peak_pct = peak_mb / self.total_vram * 100
                torch.cuda.reset_peak_memory_stats(self.device)
                vram_pct = max(vram_pct, peak_pct)
            if vram_pct > self.cfg.vram_critical_pct:
                self.oom_count += 1
                return {"loss": float("inf"), "oom": True}

        try:
            loss_main, loss_total = self._forward_and_loss(input_ids, targets, T, clr, step, total_steps, curr_seq_len)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                loss_main, loss_total = self._forward_and_loss(input_ids, targets, T, clr, step, total_steps, curr_seq_len)
            except torch.cuda.OutOfMemoryError:
                self.oom_count += 1
                if self._reduce_batch_size():
                    return {"loss": float("inf"), "oom": True, "reduced_batch": True}
                return {"loss": float("inf"), "oom": True}

        for attempt in range(2):
            try:
                scaled_loss = loss_total / self.grad_accum_steps
                scaled_loss.backward()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 0:
                    loss_main, loss_total = self._forward_and_loss(input_ids, targets, T, clr, step, total_steps, curr_seq_len)
                    continue
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

            if self._proactive_batch_check():
                return {"loss": float("inf"), "oom": True, "reduced_batch": True}

            if self.ema:
                self.ema.update(self.model)

            self.tokens_processed += B * T
            loss_val = loss_main.item()
            return {"loss": loss_val, "grad_norm": grad_norm, "oom": False}

        loss_val = loss_main.item()
        return {"loss": loss_val, "oom": False}

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
        empty_cache_every = getattr(self.cfg, "empty_cache_every", 1000)

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(
            f"Training started on {gpu_name}: "
            f"{self.cfg.format_params()} model, "
            f"batch={self.cfg.batch_size}, seq_len={self.cfg.seq_len}, "
            f"total_steps={total_steps}"
        )
        print(f"Logs: {self.log_dir / 'bulba1.jsonl'}")

        if not getattr(self.cfg, "skip_preflight", False):
            print("[PRE-FLIGHT] Running memory test...")
            preflight = preflight_memory_test(self.model, self.cfg, self.device, max_attempts=3)
            if not preflight["success"]:
                print(f"[ERROR] Pre-flight FAILED: {preflight['error']}")
                return self.model
            if preflight["batch_size"] < self.cfg.batch_size:
                old_bs = self.cfg.batch_size
                self.cfg.batch_size = preflight["batch_size"]
                self.grad_accum_steps = preflight["grad_accum"]
                print(
                    f"[PRE-FLIGHT] Adjusted: batch_size {old_bs} → {self.cfg.batch_size}, "
                    f"grad_accum → {self.grad_accum_steps}"
                )
            print("[PRE-FLIGHT] PASSED")
        else:
            print(f"[SKIP] Pre-flight disabled. Using batch_size={self.cfg.batch_size}")

        self.step_times = []
        self.step_start_time = time.time()

        step = resume_step
        while step < total_steps:
            self.step = step
            accum_loss = 0.0
            valid_steps = 0
            stage = stage_for_step(step, total_steps)

            torch.compiler.cudagraph_mark_step_begin()
            self._update_grad_accum(step, total_steps)

            curr_seq_len = self.compute_curriculum_seq_len(step, total_steps)

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
                    batch,
                    is_accum_last=is_last,
                    step=step,
                    total_steps=total_steps,
                    curr_seq_len=curr_seq_len,
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
            self.ema_loss = 0.9 * (self.ema_loss or loss_val) + 0.1 * loss_val

            if (step + 1) % checkpoint_every == 0:
                self._save_checkpoint(step, loss_val, lr, stage.name())
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if eval_every > 0 and (step + 1) % eval_every == 0:
                self.model.eval()
                if eval_loader:
                    ppl = compute_perplexity(self.model, eval_loader, self.device, cfg.eval_max_batches)
                    print(f"[EVAL] Step {step+1}: perplexity={ppl:.2f}")
                    self._log_eval(step, ppl)
                self.model.train()

            gen_every = getattr(self.cfg, "gen_every", 2000)
            if (step + 1) % gen_every == 0:
                self.model.eval()
                prompts = getattr(cfg, "gen_prompts", None) or [
                    "The capital of France is", "def fibonacci(n):", "Once upon a time",
                ]
                try:
                    samples = generate_samples(self.model, self.tokenizer, prompts, self.device, max_new_tokens=30)
                    self._log_gen(step, samples, prompts)
                except Exception:
                    pass
                self.model.train()

            if (step + 1) % log_every == 0 or step == 0:
                self._log_status(step, total_steps, loss_val, stage, lr)

            if self.autopilot is not None and self.ema_loss is not None:
                action = self.autopilot.step(step, self.ema_loss)
                if action["action"] != "none":
                    details = {k: v for k, v in action.items() if k != "action"}
                    print(f"[AUTO] {action['action']}: {details}")
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.cfg.learning_rate
                    for group in self.optimizer.param_groups:
                        group["weight_decay"] = self.cfg.weight_decay
                if action["action"] == "stop":
                    break

            step += 1

        print("[DONE] Training complete!")
        if self.ema:
            self.ema.apply_shadow(self.model)
        return self.model

    def _save_checkpoint(self, step: int, loss_val: float, lr: float, stage_name: str):
        config_meta = {"lr": lr, "stage": stage_name}
        if self.autopilot is not None:
            config_meta["autopilot"] = self.autopilot.state_dict()
        self.checkpoint_mgr.save(
            self.model,
            self.optimizer,
            step + 1,
            loss_val,
            config=config_meta,
            ema=self.ema,
        )
        print(f"[CHECKPOINT] Saved at step {step + 1}")

    # ------------------------------------------------------------------
    def resume_from_checkpoint(self, checkpoint_arg):
        if isinstance(checkpoint_arg, int) and checkpoint_arg > 0:
            path = self.checkpoint_mgr.find_latest() if checkpoint_arg == 0 else str(checkpoint_arg)
        else:
            path = str(checkpoint_arg)
        loaded_step = self.checkpoint_mgr.load(
            self.model, path, optimizer=self.optimizer, ema=self.ema
        )
        if loaded_step is None:
            loaded_step = 0
            print(f"[WARN] Could not load checkpoint: {checkpoint_arg}")
        else:
            if self.autopilot is not None and self.auto_mode:
                import json, os
                meta_path = path.replace(".safetensors", ".json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    ap_state = meta.get("config", {}).get("autopilot")
                    if ap_state:
                        self.autopilot.load_state_dict(ap_state)
                        print(f"[RESUME] Autopilot state restored")
            print(f"[RESUME] Loaded checkpoint at step {loaded_step}")
        return loaded_step

    def _log_eval(self, step: int, ppl: float):
        rec = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step + 1, "type": "eval", "perplexity": round(ppl, 2),
        }
        with open(self.log_dir / "eval.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _log_gen(self, step: int, samples: list[str], prompts: list[str]):
        with open(self.log_dir / "gen.jsonl", "a") as f:
            for p, s in zip(prompts, samples):
                rec = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": step + 1, "prompt": p, "generated": s,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

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
            flush=True,
        )
        log_file = self.log_dir / "bulba1.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()


