import os
import sys
import time
import math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

from bulba1.training.optimizer import CombinedOptimizer
from bulba1.training.ema import EMA
from bulba1.training.checkpoint import CheckpointManager
from bulba1.training.stages import stage_for_step, compute_curriculum_seq_len
from bulba1.training.chunked_ce import chunked_cross_entropy
from bulba1.training.autotuner import HardwareAutotuner
from bulba1.training.eval import run_eval
from bulba1.training.monitor import SystemMonitor, preflight_memory_test


console = Console()


class TrainingEngine:
    def __init__(self, model, cfg, tokenizer, device="cuda", tuned_config=None):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.step = 0
        self.best_loss = float("inf")
        self.ema_loss = None
        self.base_batch_size = cfg.batch_size

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        self.tuner = HardwareAutotuner(device)
        if tuned_config is not None:
            tuned = tuned_config
        elif getattr(cfg, "skip_preflight", False):
            tuned = SimpleNamespace(
                batch_size=cfg.batch_size,
                use_bf16=cfg.use_f16,
                use_gradient_checkpointing=cfg.use_gradient_checkpointing,
                optimizer_type="muon" if hasattr(cfg, "use_muon") and cfg.use_muon else "adamw",
                estimated_vram_mb=0,
            )
        else:
            actual_params = sum(p.numel() for p in model.parameters())
            tuned = self.tuner.autotune(cfg, actual_params=actual_params)
            cfg.batch_size = tuned.batch_size
            cfg.use_f16 = tuned.use_bf16
            cfg.use_gradient_checkpointing = tuned.use_gradient_checkpointing
            cfg.use_muon = tuned.optimizer_type == "muon"

        gs = getattr(cfg, "grad_accum_steps", 0)
        self.grad_accum_steps = gs if gs > 0 else max(1, cfg.batch_size)

        self.total_vram = self.tuner.total_vram_mb
        self.optimizer = self._create_optimizer(model, cfg, tuned.optimizer_type)
        use_ema = tuned.estimated_vram_mb < self.total_vram * cfg.ema_vram_threshold
        self.ema = EMA(model, decay=cfg.ema_decay) if use_ema else None
        self.checkpoint_mgr = CheckpointManager(
            cfg.checkpoint_dir, keep_top_k=cfg.checkpoint_keep_top_k
        )
        self.use_amp = tuned.use_bf16
        self.use_chunked_ce = cfg.vocab_size > cfg.auto_chunked_ce_threshold
        self.chunk_size = cfg.chunk_size
        self.optimizer_type = tuned.optimizer_type
        self.start_time = time.time()
        self.tokens_processed = 0
        self.oom_count = 0
        self.batch_reductions = 0
        self.max_batch_reductions = cfg.max_batch_reductions
        self.min_batch_size = cfg.min_batch_size
        self.monitor = SystemMonitor(self.device, interval_sec=5.0)

    def _create_optimizer(self, model, cfg, optimizer_type: str):
        if optimizer_type == "adam8bit":
            from bitsandbytes.optim import Adam8bit

            return Adam8bit(
                model.parameters(),
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        return CombinedOptimizer(model, cfg)

    def _optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def compute_curriculum_seq_len(self, step: int, total: int) -> int:
        seq_lens = getattr(self.cfg, "curriculum_seq_lens", None)
        boundaries = getattr(self.cfg, "curriculum_seq_len_boundaries", None)
        if seq_lens and len(seq_lens) > 0:
            return compute_curriculum_seq_len(
                step,
                total,
                seq_lens=seq_lens,
                boundaries=boundaries,
                target_seq_len=self.cfg.seq_len,
            )
        if self.cfg.curriculum_warmup_ratio <= 0:
            return self.cfg.seq_len
        warmup = max(1, int(total * self.cfg.curriculum_warmup_ratio))
        if step >= warmup:
            return self.cfg.seq_len
        progress = step / warmup
        return max(
            self.cfg.curriculum_start_seq_len,
            int(
                self.cfg.curriculum_start_seq_len
                + (self.cfg.seq_len - self.cfg.curriculum_start_seq_len) * progress
            ),
        )

    def compute_lr(self, step: int, total: int) -> float:
        stage = stage_for_step(step, total, boundaries=getattr(self.cfg, "stage_boundaries", None))
        base_lr = self.cfg.learning_rate * stage.lr_multiplier(
            getattr(self.cfg, "stage_lr_multipliers", None)
        )
        warmup = max(1, int(total * self.cfg.warmup_ratio))

        if step < warmup:
            return base_lr * (step / warmup)

        if getattr(self.cfg, "use_lr_cooldown", False):
            cooldown_start = int(total * (1 - self.cfg.lr_cooldown_ratio))
            if step >= cooldown_start:
                cooldown_progress = (step - cooldown_start) / max(1, total - cooldown_start)
                return base_lr * 0.01 * (1 - cooldown_progress)

        progress = (step - warmup) / max(1, total - warmup)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * max(decay, 0.01)

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

    def get_vram_usage(self) -> tuple:
        if self.device.type != "cuda":
            return 0, 0, 0.0
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        pct = allocated / self.total_vram * 100 if self.total_vram > 0 else 0
        return allocated, reserved, pct

    def _check_vram_proactive(self) -> bool:
        if self.device.type != "cuda":
            return True
        _, _, pct = self.get_vram_usage()
        if pct > self.cfg.vram_warn_pct:
            torch.cuda.empty_cache()
            _, _, pct = self.get_vram_usage()
        return pct < self.cfg.vram_critical_pct

    def _reduce_batch_size(self):
        if self.batch_reductions >= self.max_batch_reductions:
            return False
        old_bs = self.cfg.batch_size
        new_bs = max(self.min_batch_size, old_bs // 2)
        if new_bs == old_bs:
            return False
        self.cfg.batch_size = new_bs
        self.grad_accum_steps = max(1, self.grad_accum_steps * 2)
        self.batch_reductions += 1
        console.print(
            f"[yellow]OOM recovery: batch_size {old_bs} -> {new_bs}, accum_steps -> {self.grad_accum_steps}[/yellow]"
        )
        torch.cuda.empty_cache()
        return True

    def safe_forward(self, input_ids: torch.Tensor) -> tuple:
        if self.use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                return self.model(input_ids, self.cfg.checkpoint_every_n_layers)
        return self.model(input_ids, self.cfg.checkpoint_every_n_layers)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_chunked_ce:
            return chunked_cross_entropy(
                logits, targets, chunk_size=self.chunk_size, ignore_index=self.cfg.ignore_index
            )
        return F.cross_entropy(logits, targets, label_smoothing=self.cfg.label_smoothing)

    def _skip_gram_loss(self, logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or logits.size(1) <= k:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        pred = logits[:, :-k, :].reshape(-1, self.cfg.vocab_size)
        tgt = targets[:, k:].reshape(-1)
        return F.cross_entropy(pred, tgt)

    def train_step(
        self, batch, is_accum_last: bool = True, step: int = 0, total_steps: int = 1
    ) -> dict:
        self.model.train()
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, targets = batch
        else:
            input_ids = batch
            targets = batch

        # If batch_size was reduced due to OOM, slice the batch to fit.
        # The DataLoader may still yield the original larger batch size.
        if input_ids.size(0) > self.cfg.batch_size:
            input_ids = input_ids[: self.cfg.batch_size]
            targets = targets[: self.cfg.batch_size]

        curr_seq_len = self.compute_curriculum_seq_len(step, total_steps)
        if input_ids.size(1) > curr_seq_len:
            input_ids = input_ids[:, :curr_seq_len]
            targets = targets[:, :curr_seq_len]

        B, T = input_ids.shape
        clr = self.cfg.num_clr_tokens

        if not self._check_vram_proactive():
            self.oom_count += 1
            return {"loss": float("inf"), "oom": True, "vram_critical": True}

        try:
            logits, mtp1, mtp2, aux_loss = self.safe_forward(input_ids)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.oom_count += 1
            if self._reduce_batch_size():
                return {"loss": float("inf"), "oom": True, "reduced_batch": True}
            return {"loss": float("inf"), "oom": True}

        text_logits = logits[:, clr : (clr + T), :].reshape(-1, self.cfg.vocab_size)
        loss_main = self.compute_loss(text_logits, targets.reshape(-1))

        loss_total = loss_main
        metrics = {"loss_main": loss_main}

        if self.cfg.use_skip_gram and self.cfg.skip_gram_range > 1:
            for k in range(2, self.cfg.skip_gram_range + 1):
                sg_loss = self._skip_gram_loss(logits[:, clr : (clr + T), :], targets, k - 1)
                if sg_loss.item() > 0:
                    loss_total = loss_total + self.cfg.skip_gram_weight * sg_loss

        if mtp1 is not None:
            mtp1_text = mtp1[:, clr : (clr + T), :].reshape(-1, self.cfg.vocab_size)
            loss_mtp1 = self.compute_loss(mtp1_text, targets.reshape(-1))
            loss_total = loss_total + self.cfg.loss_mtp1_weight * loss_mtp1
            metrics["loss_mtp1"] = loss_mtp1

        if mtp2 is not None:
            mtp2_text = mtp2[:, clr : (clr + T), :].reshape(-1, self.cfg.vocab_size)
            loss_mtp2 = self.compute_loss(mtp2_text, targets.reshape(-1))
            loss_total = loss_total + self.cfg.loss_mtp2_weight * loss_mtp2
            metrics["loss_mtp2"] = loss_mtp2

        loss_total = loss_total + aux_loss * self.cfg.router_z_loss_coef
        metrics["aux_loss"] = aux_loss
        metrics["loss_total"] = loss_total

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
                self._optimizer_step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.oom_count += 1
                if self._reduce_batch_size():
                    return {"loss": float("inf"), "oom": True, "reduced_batch": True}
                return {"loss": float("inf"), "oom": True}

            # Track gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.float().norm()
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            metrics["grad_norm"] = total_norm

            if self.ema:
                self.ema.update(self.model)

            self.tokens_processed += B * T
            metrics["loss"] = loss_main
            return metrics

    def resume_from_checkpoint(self, path: str = "latest") -> int:
        step = self.checkpoint_mgr.load(self.model, path, optimizer=self.optimizer, ema=self.ema)
        if step:
            console.print(f"[green]Resumed from checkpoint at step {step}[/green]")
            return step
        return 0

    def _log_simple(
        self,
        step,
        total_steps,
        loss_val,
        stage,
        opt_name,
        lr,
        vram_pct,
        tok_per_sec,
        grad_norm=None,
    ):
        snap = self.monitor.snapshot(force=True)

        # Calculate ETA
        if hasattr(self, "step_times") and len(self.step_times) > 10:
            avg_time = sum(self.step_times[-10:]) / 10
            remaining_steps = total_steps - step - 1
            eta_seconds = avg_time * remaining_steps
            eta_str = f"ETA={eta_seconds / 60:.1f}m"
        else:
            eta_str = ""

        status = f"Step {step + 1}/{total_steps} | loss={loss_val:.4f} | {stage.name()} | {opt_name} | LR={lr:.2e}"
        sys_info = f"VRAM={snap.vram_reserved_mb:.0f}/{snap.vram_total_mb:.0f}MB({snap.vram_pct:.0f}%) RAM={snap.ram_used_mb:.0f}/{snap.ram_total_mb:.0f}MB({snap.ram_pct:.0f}%) CPU={snap.cpu_pct:.0f}%"

        grad_str = f"grad={grad_norm:.2f}" if grad_norm is not None else ""
        perf = f"tok/s={tok_per_sec:.0f} OOM={self.oom_count} {grad_str} {eta_str}"

        print(f"[{time.strftime('%H:%M:%S')}] {status} | {sys_info} | {perf}", flush=True)

    def train(
        self,
        train_loader,
        total_steps: int,
        eval_loader=None,
        eval_every: int = 0,
        eval_prompts=None,
        resume_step: int = 0,
        checkpoint_every: int = 50,
        log_every: int = 100,
    ):
        use_tui = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        skip_preflight = (
            getattr(self.cfg, "skip_preflight", False)
            or os.environ.get("SKIP_PREFLIGHT", "0") == "1"
        )
        if not skip_preflight:
            console.print(f"[cyan]Running pre-flight memory test...[/cyan]") if use_tui else print(
                "[PRE-FLIGHT] Testing memory..."
            )
            preflight = preflight_memory_test(self.model, self.cfg, self.device, max_attempts=3)
            if not preflight["success"]:
                msg = f"Pre-flight FAILED: {preflight['error']}. Cannot start training."
                console.print(f"[red]{msg}[/red]") if use_tui else print(f"[ERROR] {msg}")
                return self.model
            msg = f"Pre-flight PASSED: batch_size={preflight['batch_size']}, measured VRAM={preflight['measured_vram_mb']:.0f}MB"
            console.print(f"[green]{msg}[/green]") if use_tui else print(f"[OK] {msg}")
            if preflight["batch_size"] != self.cfg.batch_size:
                self.cfg.batch_size = preflight["batch_size"]
                self.grad_accum_steps = preflight["grad_accum"]
                print(
                    f"[ADJUST] batch_size={self.cfg.batch_size}, grad_accum={self.grad_accum_steps}"
                )
        else:
            print(f"[SKIP] Pre-flight disabled. Using batch_size={self.cfg.batch_size}")

        self.step_times = []
        self.step_start_time = time.time()

        if use_tui:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                TextColumn,
                BarColumn,
                TimeRemainingColumn,
            )

            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            )
        else:
            import contextlib

            progress_ctx = contextlib.nullcontext()

        with progress_ctx as progress:
            if use_tui:
                pbar_task = progress.add_task(
                    f"Training {self.cfg.format_params()} model", total=total_steps
                )

            for step in range(resume_step, total_steps):
                self.step = step
                accum_loss = 0.0
                valid_steps = 0

                stage = stage_for_step(
                    step, total_steps, boundaries=getattr(self.cfg, "stage_boundaries", None)
                )

                for accum_step in range(self.grad_accum_steps):
                    try:
                        batch = next(train_loader)
                    except StopIteration:
                        break
                    except Exception as e:
                        self._emergency_checkpoint(step, f"dataloader_error: {e}")
                        return self.model

                    if isinstance(batch, (list, tuple)):
                        try:
                            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
                        except Exception as e:
                            self._emergency_checkpoint(step, f"device_transfer_error: {e}")
                            return self.model
                    else:
                        try:
                            batch = batch.to(self.device)
                        except Exception as e:
                            self._emergency_checkpoint(step, f"device_transfer_error: {e}")
                            return self.model

                    allocated, _reserved, vram_pct = self.get_vram_usage()
                    if vram_pct > self.cfg.vram_critical_pct:
                        self._emergency_checkpoint(step, f"vram_critical: {vram_pct:.1f}%")
                        return self.model

                    self.update_bitnet_activation_bits(step, total_steps)

                    lr = self.compute_lr(step, total_steps)
                    if hasattr(self.optimizer, "param_groups"):
                        for group in self.optimizer.param_groups:
                            group["lr"] = lr
                    elif hasattr(self.optimizer, "muon") and self.optimizer.muon:
                        for group in self.optimizer.muon.param_groups:
                            group["lr"] = lr
                    if hasattr(self.optimizer, "sgd") and self.optimizer.sgd:
                        for group in self.optimizer.sgd.param_groups:
                            group["lr"] = lr

                    is_last = accum_step == self.grad_accum_steps - 1
                    try:
                        metrics = self.train_step(
                            batch, is_accum_last=is_last, step=step, total_steps=total_steps
                        )
                    except Exception as e:
                        print(f"[ERROR] train_step failed at step {step}: {e}", flush=True)
                        self._emergency_checkpoint(step, f"train_step_error: {e}")
                        return self.model

                if metrics.get("reduced_batch"):
                    self.optimizer.zero_grad()
                    valid_steps = 0
                    break
                if metrics.get("oom"):
                    continue
                accum_loss += metrics["loss"].item()
                valid_steps += 1

                if valid_steps == 0:
                    continue

                loss_val = accum_loss / valid_steps
                if loss_val < self.best_loss:
                    self.best_loss = loss_val

                self.ema_loss = 0.9 * (self.ema_loss or loss_val) + 0.1 * loss_val

                if self.optimizer_type == "adam8bit":
                    opt_name = "AdamW 8-bit"
                elif self.optimizer_type == "adamw":
                    opt_name = "AdamW"
                else:
                    opt_name = "Muon"

                if use_tui:
                    progress.update(
                        pbar_task,
                        advance=1,
                        description=f"Step {step + 1}/{total_steps} | loss={loss_val:.4f} | {stage.name()} | {opt_name}",
                    )

                if eval_every > 0 and (step + 1) % eval_every == 0 and eval_loader is not None:
                    try:
                        if use_tui:
                            console.print(f"[cyan]Running eval at step {step + 1}...[/cyan]")
                        run_eval(self.model, self.tokenizer, eval_loader, self.device, eval_prompts)
                    except Exception as e:
                        if use_tui:
                            console.print(f"[yellow]Eval error: {e}. Continuing.[/yellow]")

                if (step + 1) % log_every == 0:
                    elapsed = time.time() - self.start_time
                    tok_per_sec = self.tokens_processed / elapsed if elapsed > 0 else 0
                    # Build full config for checkpoint (for resume verification)
                    train_config = {
                        "lr": lr,
                        "vram_pct": vram_pct,
                        "tok/s": tok_per_sec,
                        "stage": stage.name(),
                        "d_model": self.cfg.d_model,
                        "n_layers": self.cfg.n_layers,
                        "n_heads": self.cfg.n_heads,
                        "vocab_size": self.cfg.vocab_size,
                        "seq_len": self.cfg.seq_len,
                        "batch_size": self.cfg.batch_size,
                        "use_mamba": self.cfg.use_mamba,
                        "use_bitlinear": self.cfg.use_bitlinear,
                        "use_moe": self.cfg.use_moe,
                        "num_experts": self.cfg.num_experts,
                        "expert_hidden": self.cfg.expert_hidden,
                    }
                    self.checkpoint_mgr.save(
                        self.model,
                        self.optimizer,
                        step + 1,
                        loss_val,
                        config=train_config,
                        ema=self.ema,
                    )
                    if use_tui:
                        snap = self.monitor.snapshot(force=True)
                        table = Table(title=f"Step {step + 1}/{total_steps} — {stage.name()}")
                        table.add_column("Metric", style="cyan")
                        table.add_column("Value", style="magenta")
                        table.add_row("Loss", f"{loss_val:.4f}")
                        table.add_row("EMA Loss", f"{self.ema_loss:.4f}")
                        table.add_row("Best Loss", f"{self.best_loss:.4f}")
                        table.add_row("LR", f"{lr:.2e}")
                        table.add_row("Stage", stage.name())
                        table.add_row("Optimizer", opt_name)
                        table.add_row(
                            "VRAM",
                            f"{snap.vram_reserved_mb:.0f}/{snap.vram_total_mb:.0f}MB ({snap.vram_pct:.1f}%)",
                        )
                        table.add_row(
                            "RAM",
                            f"{snap.ram_used_mb:.0f}/{snap.ram_total_mb:.0f}MB ({snap.ram_pct:.1f}%)",
                        )
                        if snap.gpu_util_pct is not None:
                            table.add_row("GPU", f"{snap.gpu_util_pct:.0f}%")
                        if snap.gpu_temp_c is not None:
                            table.add_row("GPU Temp", f"{snap.gpu_temp_c:.0f}C")
                        table.add_row("CPU", f"{snap.cpu_pct:.0f}%")
                        table.add_row("Tokens/s", f"{tok_per_sec:.0f}")
                        table.add_row("OOM skips", str(self.oom_count))
                        table.add_row("Batch size", str(self.cfg.batch_size))
                        if self.use_amp:
                            table.add_row("Precision", "BF16")
                        console.print(table)
                    else:
                        grad_norm = metrics.get("grad_norm")
                        self._log_simple(
                            step,
                            total_steps,
                            loss_val,
                            stage,
                            opt_name,
                            lr,
                            vram_pct,
                            tok_per_sec,
                            grad_norm,
                        )

                    step_time = time.time() - self.step_start_time
                    self.step_times.append(step_time)
                    if len(self.step_times) > 100:
                        self.step_times = self.step_times[-100:]
                    self.step_start_time = time.time()

        if use_tui:
            console.print("[green]Training complete![/green]")
        else:
            print("[DONE] Training complete!")
        if self.ema:
            self.ema.apply_shadow(self.model)
        return self.model

    def _emergency_checkpoint(self, step: int, reason: str):
        try:
            self.checkpoint_mgr.save(
                self.model,
                self.optimizer,
                step,
                getattr(self, "ema_loss", float("inf")),
                config={"reason": reason, "paused": True},
                ema=self.ema,
            )
            console.print(
                f"[yellow]Emergency checkpoint saved at step {step}. Reason: {reason}[/yellow]"
            )
            console.print(f"[yellow]Resume with: python -m bulba1.cli --resume ...[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to save emergency checkpoint: {e}[/red]")
