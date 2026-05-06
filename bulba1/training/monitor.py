import os
import time
import psutil
import torch
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SystemSnapshot:
    timestamp: float
    ram_used_mb: float
    ram_total_mb: float
    ram_pct: float
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_total_mb: float
    vram_pct: float
    cpu_pct: float
    gpu_util_pct: Optional[float] = None
    gpu_temp_c: Optional[float] = None


class SystemMonitor:
    """Continuously monitor RAM, VRAM, CPU, and GPU utilization."""

    def __init__(self, device: torch.device, interval_sec: float = 5.0):
        self.device = device
        self.interval_sec = interval_sec
        self.total_ram_mb = psutil.virtual_memory().total / 1024 / 1024
        self.total_vram_mb = 0.0
        self.has_gpu = device.type == "cuda"
        if self.has_gpu:
            try:
                self.total_vram_mb = (
                    torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
                )
            except Exception:
                self.has_gpu = False
        self._history: list = []
        self._last_check = 0.0

    def snapshot(self, force: bool = False) -> SystemSnapshot:
        now = time.time()
        if not force and now - self._last_check < self.interval_sec and self._history:
            return self._history[-1]
        self._last_check = now

        ram = psutil.virtual_memory()
        ram_used_mb = ram.used / 1024 / 1024
        ram_pct = ram.percent

        cpu_pct = psutil.cpu_percent(interval=None)

        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        vram_pct = 0.0
        gpu_util_pct = None
        gpu_temp_c = None

        if self.has_gpu:
            try:
                vram_allocated_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                vram_reserved_mb = torch.cuda.memory_reserved(self.device) / 1024 / 1024

                try:
                    import pynvml

                    if not hasattr(self, "_nvml_initialized"):
                        pynvml.nvmlInit()
                        self._nvml_initialized = True
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_reserved_mb = mem_info.used / 1024 / 1024
                    vram_pct = (
                        (vram_reserved_mb / self.total_vram_mb) * 100
                        if self.total_vram_mb > 0
                        else 0
                    )

                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_temp_c = temp
                    except Exception:
                        pass
                except ImportError:
                    vram_pct = (
                        (vram_reserved_mb / self.total_vram_mb) * 100
                        if self.total_vram_mb > 0
                        else 0
                    )
            except Exception:
                pass

        snap = SystemSnapshot(
            timestamp=now,
            ram_used_mb=ram_used_mb,
            ram_total_mb=self.total_ram_mb,
            ram_pct=ram_pct,
            vram_allocated_mb=vram_allocated_mb,
            vram_reserved_mb=vram_reserved_mb,
            vram_total_mb=self.total_vram_mb,
            vram_pct=vram_pct,
            cpu_pct=cpu_pct,
            gpu_util_pct=gpu_util_pct,
            gpu_temp_c=gpu_temp_c,
        )
        self._history.append(snap)
        # Keep last 100 snapshots
        if len(self._history) > 100:
            self._history = self._history[-100:]
        return snap

    def format_status(self, snap: Optional[SystemSnapshot] = None) -> str:
        if snap is None:
            snap = self.snapshot()
        parts = [
            f"RAM {snap.ram_used_mb:.0f}/{snap.ram_total_mb:.0f}MB ({snap.ram_pct:.0f}%)",
        ]
        if self.has_gpu:
            parts.append(
                f"VRAM {snap.vram_reserved_mb:.0f}/{snap.vram_total_mb:.0f}MB ({snap.vram_pct:.0f}%)"
            )
            if snap.gpu_util_pct is not None:
                parts.append(f"GPU {snap.gpu_util_pct:.0f}%")
            if snap.gpu_temp_c is not None:
                parts.append(f"Temp {snap.gpu_temp_c:.0f}°C")
        parts.append(f"CPU {snap.cpu_pct:.0f}%")
        return " | ".join(parts)

    def check_critical(self, cfg) -> Dict[str, bool]:
        snap = self.snapshot()
        return {
            "ram_critical": snap.ram_pct > 95,
            "vram_critical": snap.vram_pct > getattr(cfg, "vram_critical_pct", 95.0),
            "vram_warn": snap.vram_pct > getattr(cfg, "vram_warn_pct", 88.0),
            "gpu_hot": (snap.gpu_temp_c or 0) > 85,
        }

    def get_history(self) -> list:
        return self._history


def preflight_memory_test(model, cfg, device, max_attempts: int = 3) -> dict:
    """Test forward+backward with dummy batch before real training.

    Returns dict with:
        - success: bool
        - measured_vram_mb: float (peak VRAM during test)
        - batch_size: int (adjusted batch size)
        - grad_accum: int (adjusted grad accum)
        - error: str (if failed)
    """
    import torch
    from torch.amp import autocast

    def _test_once(batch_size: int, seq_len: int) -> float:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_vram = torch.cuda.memory_allocated(device) / 1024 / 1024

        dummy_input = torch.randint(
            0,
            cfg.vocab_size,
            (batch_size, seq_len),
            device=device,
        )

        amp_ctx = (
            autocast("cuda", dtype=torch.bfloat16)
            if getattr(cfg, "use_f16", True)
            else torch.no_grad()
        )
        with amp_ctx:
            logits, mtp1, mtp2, aux_loss = model(dummy_input, cfg.checkpoint_every_n_layers)

        loss = logits.mean()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            return peak_vram - base_vram
        return 0.0

    model.eval()
    batch_size = cfg.batch_size
    grad_accum = getattr(cfg, "grad_accum_steps", max(1, cfg.batch_size))
    seq_len = cfg.seq_len

    for attempt in range(max_attempts):
        try:
            measured_vram = _test_once(batch_size, seq_len)
            model.train()
            return {
                "success": True,
                "measured_vram_mb": measured_vram,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "attempts": attempt + 1,
            }
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            old_bs = batch_size
            batch_size = max(1, batch_size // 2)
            grad_accum = max(1, grad_accum * 2)
            if batch_size == old_bs:
                model.train()
                return {
                    "success": False,
                    "measured_vram_mb": 0.0,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "attempts": attempt + 1,
                    "error": f"OOM at batch_size={old_bs}, cannot reduce further",
                }

    model.train()
    return {
        "success": False,
        "measured_vram_mb": 0.0,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "attempts": max_attempts,
        "error": f"OOM after {max_attempts} attempts, final batch_size={batch_size}",
    }
