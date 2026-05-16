# Bulba 1 — Complete Codebase

Generated: 2026-05-16 17:12

## Contents

- bulba1/autonomy.py
- bulba1/cli.py
- bulba1/config.py
- bulba1/model/bit_linear.py
- bulba1/model/block.py
- bulba1/model/diff_attn.py
- bulba1/model/kda.py
- bulba1/model/mamba.py
- bulba1/model/mhc.py
- bulba1/model/minichat.py
- bulba1/model/mod.py
- bulba1/model/moe.py
- bulba1/model/token_merging.py
- bulba1/orchestrator.py
- bulba1/tokenizer.py
- bulba1/training/checkpoint.py
- bulba1/training/chunked_ce.py
- bulba1/training/ema.py
- bulba1/training/engine.py
- bulba1/training/eval.py
- bulba1/training/monitor.py
- bulba1/training/optimizer.py
- bulba1/training/stages.py
- configs/auto.yaml
- configs/default.yaml
- configs/e2e_test.yaml
- configs/smoke_test.yaml
- docs/ARCHITECTURE.md
- docs/COMPONENTS.md
- docs/CONFIG_GUIDE.md
- docs/DEVELOPER_GUIDE.md
- docs/PAPERS.md
- docs/TRAINING.md
- scripts/auto_config.py
- scripts/build_and_tokenize.py
- scripts/download_all_datasets.py
- scripts/dpo_train.py
- scripts/pretokenize.py
- scripts/sft_train.py
- tests/test_sanity.py
- tools/deep_profile.py
- tools/log_viz.py

---

## bulba1/autonomy.py

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Mode(Enum):
    CALIBRATE = "calibrate"
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    OSCILLATE = "oscillate"
    PLATEAU = "plateau"
    SGDR = "sgdr"


@dataclass
class PilotState:
    mode: Mode = Mode.CALIBRATE
    stopped: bool = False

    current_lr: float = 0.0
    current_wd: float = 0.0
    current_noise: float = 0.0
    current_sd: float = 0.0
    stable_lr: float = 0.0
    max_stable_lr: float = 0.0

    plateau_level: int = 0
    oscillation_strikes: int = 0

    sgdr_cycle: int = 0
    sgdr_step: int = 0
    sgdr_T: int = 0

    intervention_active: bool = False
    intervention_step: int = 0
    intervention_snapshot: dict = field(default_factory=dict)

    ema_window: list[float] = field(default_factory=list)
    calibrated: bool = False
    base_vol: float = 0.0
    base_improvement: float = 0.0
    boost_multiplier: float = 0.05


class AutoPilot:
    def __init__(self, cfg, log_dir: str = "logs", confidence: float = 0.95):
        self.cfg = cfg
        self.confidence = confidence

        self.total_steps = getattr(cfg, "total_steps", 25000)
        self.base_lr = cfg.learning_rate
        self.base_wd = cfg.weight_decay
        self.base_noise = getattr(cfg, "gradient_noise", 0.0)
        self.base_sd = getattr(cfg, "stochastic_depth_prob", 0.0)

        self.warmup_steps = max(1, int(self.total_steps * getattr(cfg, "warmup_ratio", 0.05)))
        self._derive_timings()

        self.state = PilotState(
            mode=Mode.CALIBRATE,
            current_lr=self.base_lr,
            current_wd=self.base_wd,
            current_noise=self.base_noise,
            current_sd=self.base_sd,
            stable_lr=self.base_lr,
            max_stable_lr=self.base_lr,
        )

        self.log_path = Path(log_dir) / "autonomy.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_check = -1
        self._action_seq = 0

    def _derive_timings(self):
        T = self.total_steps
        self.trend_window = max(30, T // 200)
        self.vol_window = max(10, T // 800)
        self.check_every = max(10, T // 500)
        self.calibrate_steps = max(100, T // 100)
        self.intervention_patience = max(100, T // 80)
        self.sgdr_base_T = max(200, T // 10)
        self.min_plateau_steps = max(300, T // 50)
        self.max_oscillation_strikes = 3

    def compute_lr(self, step: int) -> float:
        if self.state.stopped:
            return 0.0
        if step < self.warmup_steps:
            return self.base_lr * (step / max(1, self.warmup_steps))
        if self.state.mode == Mode.SGDR and self.state.sgdr_T > 0:
            t = min(self.state.sgdr_step, self.state.sgdr_T)
            lo = self.state.current_lr * 0.01
            hi = self.state.current_lr
            return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * t / self.state.sgdr_T))
        if getattr(self.cfg, "use_inv_sqrt_lr", False) and self.state.mode in (Mode.EXPLORE, Mode.EXPLOIT):
            return self.state.current_lr * (self.warmup_steps / max(step, self.warmup_steps)) ** 0.5
        return self.state.current_lr

    def step(self, step: int, ema: float) -> dict:
        if self.state.stopped:
            return {"action": "none"}

        self.state.ema_window.append(ema)
        cap = max(self.trend_window * 3, self.calibrate_steps * 2)
        if len(self.state.ema_window) > cap:
            self.state.ema_window = self.state.ema_window[-cap:]

        if self.state.mode == Mode.SGDR:
            self.state.sgdr_step += 1
            if self.state.sgdr_step >= self.state.sgdr_T:
                return self._sgdr_cycle_end(step)
            return {"action": "none"}

        if self.state.intervention_active:
            return self._check_intervention(step)

        if step - self._last_check < self.check_every:
            return {"action": "none"}

        if self.state.mode == Mode.CALIBRATE:
            if step >= self.warmup_steps + self.calibrate_steps:
                self._finish_calibration()
                self.state.mode = Mode.EXPLORE
                self._last_check = step
                self._write_cfg()
                return self._act(step, "M_explore", from_mode="calibrate",
                                 base_vol=f"{self.state.base_vol:.6f}",
                                 base_imp=f"{self.state.base_improvement:.6f}")
            return {"action": "none"}

        self._last_check = step
        rel_range = self._relative_range()
        rel_trend = self._relative_trend()

        handlers = {
            Mode.EXPLORE:    lambda: self._handle_explore(step, rel_range, rel_trend),
            Mode.EXPLOIT:    lambda: self._handle_exploit(step, rel_range, rel_trend),
            Mode.OSCILLATE:  lambda: self._handle_oscillate(step, rel_range, rel_trend),
            Mode.PLATEAU:    lambda: self._handle_plateau(step, rel_range, rel_trend),
        }
        h = handlers.get(self.state.mode)
        return h() if h else {"action": "none"}

    def _finish_calibration(self):
        w = self.state.ema_window
        if len(w) < 20:
            return
        n = len(w)
        deltas = [abs(w[i] - w[i - 1]) / max(abs(w[i - 1]), 1e-10) for i in range(1, n)]
        deltas.sort()
        self.state.base_vol = deltas[int(len(deltas) * 0.85)]
        improvements = [(w[0] - w[-1]) / max(abs(w[0]), 1e-10) / n]
        self.state.base_improvement = max(0, improvements[0])
        self.state.calibrated = True

    def _relative_range(self) -> float:
        w = self.state.ema_window[-self.vol_window:]
        if len(w) < 5:
            return 0.0
        mn, mx = min(w), max(w)
        return (mx - mn) / max(abs((mx + mn) / 2), 1e-10)

    def _relative_trend(self) -> float:
        w = self.state.ema_window[-self.trend_window:]
        n = len(w)
        if n < 10:
            return 0.0
        ym = sum(w) / n
        xm = (n - 1) / 2.0
        num = sum((i - xm) * (v - ym) for i, v in enumerate(w))
        den = sum((i - xm) ** 2 for i in range(n))
        return (num / max(den, 1)) / max(abs(ym), 1e-10)

    def _vol_ok(self, rel_range: float) -> bool:
        if not self.state.calibrated or self.state.base_vol == 0:
            return rel_range < 0.015
        return rel_range < self.state.base_vol * 5.0

    def _trend_flat(self, rel_trend: float) -> bool:
        if not self.state.calibrated or self.state.base_improvement == 0:
            return abs(rel_trend) < 1e-4
        return abs(rel_trend) < self.state.base_improvement * 0.3

    def _trend_down(self, rel_trend: float) -> bool:
        if not self.state.calibrated or self.state.base_improvement == 0:
            return rel_trend < -1e-4
        return rel_trend < -self.state.base_improvement * 0.25

    def _trend_strong_down(self, rel_trend: float) -> bool:
        if not self.state.calibrated or self.state.base_improvement == 0:
            return rel_trend < -5e-4
        return rel_trend < -self.state.base_improvement * 1.5

    def _handle_explore(self, step: int, rr: float, rt: float) -> dict:
        if not self._vol_ok(rr):
            self.state.stable_lr = self.state.current_lr
            self.state.max_stable_lr = max(self.state.max_stable_lr, self.state.current_lr)
            return self._trans(step, Mode.EXPLOIT, lr=self.state.current_lr * 0.75)

        if step > self.min_plateau_steps and self._trend_flat(rt):
            return self._trans(step, Mode.PLATEAU)

        if self._trend_down(rt):
            new_lr = self.state.current_lr * (1.0 + self.state.boost_multiplier)
            if self.state.current_lr > self.state.max_stable_lr:
                self.state.max_stable_lr = self.state.current_lr
            return self._go(step, "boost_lr", lr=new_lr)

        return {"action": "none"}

    def _handle_exploit(self, step: int, rr: float, rt: float) -> dict:
        if not self._vol_ok(rr):
            self.state.oscillation_strikes += 1
            if self.state.oscillation_strikes >= self.max_oscillation_strikes:
                self.state.oscillation_strikes = 0
                return self._trans(step, Mode.SGDR,
                    base_lr=self.state.stable_lr * 0.7, sgdr_T=self.sgdr_base_T)
            return self._trans(step, Mode.OSCILLATE, lr=self.state.current_lr * 0.7)
        self.state.oscillation_strikes = max(0, self.state.oscillation_strikes - 1)

        if self._trend_flat(rt):
            return self._trans(step, Mode.PLATEAU)

        if self._trend_strong_down(rt):
            self.state.boost_multiplier = min(0.25, self.state.boost_multiplier * 1.2)
            return self._trans(step, Mode.EXPLORE)

        return {"action": "none"}

    def _handle_oscillate(self, step: int, rr: float, rt: float) -> dict:
        if self._vol_ok(rr):
            self.state.stable_lr = self.state.current_lr
            self.state.max_stable_lr = max(self.state.max_stable_lr, self.state.current_lr)
            self.state.boost_multiplier = max(0.02, self.state.boost_multiplier * 0.7)
            self.state.oscillation_strikes = 0
            return self._trans(step, Mode.EXPLOIT)
        self.state.oscillation_strikes += 1
        if self.state.oscillation_strikes >= self.max_oscillation_strikes:
            self.state.oscillation_strikes = 0
            return self._trans(step, Mode.SGDR,
                base_lr=self.state.stable_lr * 0.7, sgdr_T=self.sgdr_base_T)
        return self._go(step, "osc_cut", lr=self.state.current_lr * 0.7)

    def _handle_plateau(self, step: int, rr: float, rt: float) -> dict:
        if self._trend_strong_down(rt):
            self.state.plateau_level = 0
            if self.state.boost_multiplier < 0.05:
                self.state.boost_multiplier = 0.05
            return self._trans(step, Mode.EXPLORE)

        stages = [
            ("reduce_lr",      {"lr": self.state.current_lr * 0.5}),
            ("tighten_reg",    {"lr": self.state.current_lr * 0.3, "wd": self.state.current_wd * 2}),
            ("add_noise",      {"noise": self.state.current_noise * 3 + 1e-5}),
            ("stoch_depth",    {"sd": min(self.state.current_sd * 2, 0.35)}),
        ]

        if self.state.plateau_level >= len(stages):
            return self._trans(step, Mode.SGDR,
                base_lr=self.state.stable_lr * 0.7, sgdr_T=self.sgdr_base_T)

        name, changes = stages[self.state.plateau_level]
        self.state.plateau_level += 1
        return self._go(step, name, **changes)

    def _go(self, step: int, name: str, **ch) -> dict:
        snap = {"lr": self.state.current_lr, "wd": self.state.current_wd,
                "noise": self.state.current_noise, "sd": self.state.current_sd}
        self._apply_changes(ch)
        self.state.intervention_active = True
        self.state.intervention_step = step
        self.state.intervention_snapshot = snap
        self._write_cfg()
        return self._act(step, name, **ch)

    def _trans(self, step: int, new_mode: Mode, **ch) -> dict:
        old = self.state.mode.value
        ch.pop("from_mode", None)
        self.state.mode = new_mode
        self.state.ema_window.clear()
        self.state.intervention_active = False
        self._last_check = step
        self._apply_changes(ch)
        if "sgdr_T" in ch:
            self.state.sgdr_T = ch.pop("sgdr_T")
            self.state.sgdr_step = 0
            self.state.sgdr_cycle = 0
        if "base_lr" in ch:
            self.state.current_lr = ch.pop("base_lr")
        self._write_cfg()
        return self._act(step, f"M_{new_mode.value}", from_mode=old, **ch)

    def _check_intervention(self, step: int) -> dict:
        if step - self.state.intervention_step < self.intervention_patience:
            return {"action": "none"}
        self.state.intervention_active = False

        w = self.state.ema_window
        if len(w) < 60:
            return {"action": "none"}
        r = sum(w[-30:]) / 30
        p = sum(w[:30]) / 30
        if p > 0 and r > p * 1.01:
            s = self.state.intervention_snapshot
            self.state.current_lr = s["lr"]
            self.state.current_wd = s["wd"]
            self.state.current_noise = s["noise"]
            self.state.current_sd = s["sd"]
            self.state.ema_window.clear()
            self._last_check = step
            if self.state.mode == Mode.PLATEAU:
                self.state.plateau_level += 1
            self._write_cfg()
            return self._act(step, "revert", p=f"{p:.4f}", r=f"{r:.4f}")
        return {"action": "none"}

    def _sgdr_cycle_end(self, step: int) -> dict:
        max_c = 3
        self.state.sgdr_cycle += 1
        if self.state.sgdr_cycle >= max_c:
            self.state.sgdr_T = 0
            self.state.current_lr = self.state.stable_lr
            self.state.ema_window.clear()
            self._write_cfg()
            return self._trans(step, Mode.EXPLOIT, from_mode="sgdr")
        self.state.sgdr_T = self.sgdr_base_T * (2 ** self.state.sgdr_cycle)
        self.state.sgdr_step = 0
        self.state.current_lr = max(1e-7, self.state.current_lr * 0.7)
        self._write_cfg()
        return self._act(step, "sgdr_cycle", c=self.state.sgdr_cycle, T=self.state.sgdr_T)

    def _apply_changes(self, ch: dict):
        for k, attr in [("lr", "current_lr"), ("wd", "current_wd"),
                         ("noise", "current_noise"), ("sd", "current_sd")]:
            if k in ch:
                setattr(self.state, attr, ch[k])

    def _write_cfg(self):
        for attr, val in [("learning_rate", self.state.current_lr),
                           ("weight_decay", self.state.current_wd),
                           ("gradient_noise", self.state.current_noise),
                           ("stochastic_depth_prob", self.state.current_sd)]:
            if hasattr(self.cfg, attr):
                setattr(self.cfg, attr, val)

    def _act(self, step: int, name: str, **extra) -> dict:
        self._action_seq += 1
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step, "seq": self._action_seq, "action": name,
            "mode": self.state.mode.value,
            "lr": round(self.state.current_lr, 8),
            "wd": round(self.state.current_wd, 6),
            "noise": self.state.current_noise,
            "sd": round(self.state.current_sd, 4),
            "stable_lr": round(self.state.max_stable_lr, 8),
            "boost": round(self.state.boost_multiplier, 4),
            **extra,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        return {"action": name, **extra}

    def state_dict(self) -> dict:
        return {
            "mode": self.state.mode.value,
            "stopped": self.state.stopped,
            "current_lr": self.state.current_lr,
            "current_wd": self.state.current_wd,
            "current_noise": self.state.current_noise,
            "current_sd": self.state.current_sd,
            "stable_lr": self.state.stable_lr,
            "max_stable_lr": self.state.max_stable_lr,
            "plateau_level": self.state.plateau_level,
            "oscillation_strikes": self.state.oscillation_strikes,
            "sgdr_cycle": self.state.sgdr_cycle,
            "sgdr_step": self.state.sgdr_step,
            "sgdr_T": self.state.sgdr_T,
            "intervention_active": self.state.intervention_active,
            "intervention_step": self.state.intervention_step,
            "base_vol": self.state.base_vol,
            "base_improvement": self.state.base_improvement,
            "boost_multiplier": self.state.boost_multiplier,
            "_action_seq": self._action_seq,
        }

    def load_state_dict(self, d: dict):
        try: self.state.mode = Mode(d.get("mode", "explore"))
        except ValueError: self.state.mode = Mode.EXPLORE
        self.state.stopped = d.get("stopped", False)
        self.state.current_lr = d.get("current_lr", self.base_lr)
        self.state.current_wd = d.get("current_wd", self.base_wd)
        self.state.current_noise = d.get("current_noise", self.base_noise)
        self.state.current_sd = d.get("current_sd", self.base_sd)
        self.state.stable_lr = d.get("stable_lr", self.base_lr)
        self.state.max_stable_lr = d.get("max_stable_lr", self.base_lr)
        self.state.plateau_level = d.get("plateau_level", 0)
        self.state.oscillation_strikes = d.get("oscillation_strikes", 0)
        self.state.sgdr_cycle = d.get("sgdr_cycle", 0)
        self.state.sgdr_step = d.get("sgdr_step", 0)
        self.state.sgdr_T = d.get("sgdr_T", 0)
        self.state.intervention_active = d.get("intervention_active", False)
        self.state.intervention_step = d.get("intervention_step", 0)
        self.state.calibrated = True
        self.state.base_vol = d.get("base_vol", 0.0)
        self.state.base_improvement = d.get("base_improvement", 0.0)
        self.state.boost_multiplier = d.get("boost_multiplier", 0.05)
        self._action_seq = d.get("_action_seq", 0)
        self._write_cfg()


## bulba1/cli.py

#!/usr/bin/env python3
"""
Bulba 1 — Autonomous LLM Training (YAML‑only configuration)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.tokenizer import FastTokenizer, HFTokenizer, create_dataloader
from bulba1.training.engine import TrainingEngine

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Bulba 1 — Autonomous LLM Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-prompts", type=str, nargs="*", default=[])
    parser.add_argument("--auto", action="store_true", help="Автономный режим: AutoLR + плато-детект + warm restart")
    parser.add_argument(
        "--full", action="store_true", help="Запустить полный цикл: загрузка, токенизация, обучение"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Пропустить загрузку датасетов (для --full)"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Пропустить токенизацию (для --full)"
    )
    args = parser.parse_args()

    # Полный пайплайн
    if args.full:
        from bulba1.orchestrator import BulbaOrchestrator

        orch = BulbaOrchestrator(config_path=args.config)
        orch.run_full(skip_download=args.skip_download, skip_build=args.skip_build)
        return

    # Загружаем YAML
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    all_params = {}
    all_params.update(yaml_cfg.get("model", {}))
    all_params.update(yaml_cfg.get("training", {}))
    if args.auto and "autonomy" in yaml_cfg:
        class AutonomyConfig:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        all_params["autonomy"] = AutonomyConfig(**yaml_cfg["autonomy"])
    cfg = ModelConfig(**all_params)
    print(f"📄 Загружен конфиг из {args.config}")

    # Устройство
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Устройство: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    data_dir = getattr(cfg, "data_dir", "data/tokenized")
    batch_size = getattr(cfg, "batch_size", 1)
    seq_len = getattr(cfg, "seq_len", 512)

    # Токенизатор
    if os.path.exists("data/tokenizer_fast.json"):
        tokenizer = FastTokenizer("data/tokenizer_fast.json")
        tokenizer.load()
    else:
        tokenizer = HFTokenizer(vocab_size=getattr(cfg, "vocab_size", 26000))
        if not os.path.exists(tokenizer.model_path):
            files = list(Path(data_dir).rglob("*.txt"))
            if not files:
                sys.exit("❌ Нет .txt файлов в data/train")
            tokenizer.train([str(f) for f in files])
        else:
            tokenizer.load()
    print(f"🔤 Vocab size: {tokenizer.vocab_size}")

    # DataLoader
    num_workers = getattr(cfg, "num_workers", 2)
    prefetch_factor = getattr(cfg, "prefetch_factor", 4)
    loader = create_dataloader(
        tokenizer,
        data_dir,
        batch_size,
        seq_len,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    def infinite_loader():
        while True:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    yield tuple(b.to(device, non_blocking=True) for b in batch)
                else:
                    yield batch.to(device, non_blocking=True)

    # Модель
    print("🏗️  Создание модели...")
    model = MiniChat(cfg).to(device)
    if getattr(cfg, "use_f16", False):
        model = model.to(torch.bfloat16)
        print("🔧 Модель приведена к bfloat16")
    should_compile = args.compile or getattr(cfg, "compile", False)
    if should_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("⚡ torch.compile включён")

    engine = TrainingEngine(model, cfg, tokenizer, device=str(device), auto_mode=args.auto)

    resume_step = 0
    if args.resume or args.checkpoint > 0:
        checkpoint_arg = args.checkpoint if args.checkpoint > 0 else "latest"
        resume_step = engine.resume_from_checkpoint(checkpoint_arg)
    elif os.path.exists(os.path.join(cfg.checkpoint_dir, "checkpoint_step_1000.safetensors")):
        resume_step = engine.resume_from_checkpoint("latest")
        if resume_step > 0:
            print(f"[AUTO-RESUME] Restored from step {resume_step}")

    print(f"🚀 Старт обучения на {cfg.total_steps} шагов...")
    model = engine.train(
        infinite_loader(),
        eval_loader=None,
        eval_prompts=args.eval_prompts if args.eval_prompts else None,
        resume_step=resume_step,
    )

    # Auto SFT after main training
    if getattr(cfg, "auto_sft", False):
        sft_data = getattr(cfg, "auto_sft_data", "data/sft")
        sft_data_file = os.path.join(sft_data, "sft_claude_opus47.jsonl")
        sft_epochs = getattr(cfg, "auto_sft_epochs", 3)
        sft_lr = getattr(cfg, "auto_sft_lr", 1.0e-5)
        
        print(f"\n🎯 Запуск SFT (data={sft_data_file}, epochs={sft_epochs}, lr={sft_lr})...")
        
        import subprocess
        sft_cmd = [
            sys.executable, "scripts/sft_train.py",
            "--data", sft_data_file,
            "--output", "checkpoints/sft",
            "--epochs", str(sft_epochs),
            "--lr", str(sft_lr)
        ]
        subprocess.run(sft_cmd, check=True)
        print(f"✅ SFT завершён!")

    # Auto DPO after SFT
    if getattr(cfg, "auto_dpo", False):
        dpo_data = getattr(cfg, "auto_dpo_data", "data/dpo")
        dpo_data_file = os.path.join(dpo_data, "train.jsonl")
        dpo_epochs = getattr(cfg, "auto_dpo_epochs", 3)
        dpo_lr = getattr(cfg, "auto_dpo_lr", 1.0e-6)
        dpo_beta = getattr(cfg, "auto_dpo_beta", 0.1)
        
        print(f"\n🎯 Запуск DPO (data={dpo_data_file}, epochs={dpo_epochs}, lr={dpo_lr}, beta={dpo_beta})...")
        
        import subprocess
        dpo_cmd = [
            sys.executable, "scripts/dpo_train.py",
            "--data", dpo_data_file,
            "--output", "checkpoints/dpo",
            "--epochs", str(dpo_epochs),
            "--lr", str(dpo_lr),
            "--beta", str(dpo_beta)
        ]
        subprocess.run(dpo_cmd, check=True)
        print(f"✅ DPO завершён!")

    print("\n🏁 Обучение завершено!")



if __name__ == "__main__":
    main()




## bulba1/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Minimal config – all values must be provided via YAML."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def format_params(self, n: Optional[int] = None) -> str:
        n = n or 0
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.1f}K"

## bulba1/model/bit_linear.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_b158(w: torch.Tensor) -> torch.Tensor:
    gamma = w.abs().mean().clamp_min(1e-8)
    w_norm = w / gamma
    w_quant = torch.round(torch.clamp(w_norm, -1.0, 1.0))
    w_ste = (w_quant - w_norm).detach() + w_norm
    return w_ste * gamma


def activation_quant_ste(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    Q_b = 2 ** (num_bits - 1)
    if num_bits == 8:
        gamma = x.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8)
        scale = gamma / (Q_b - 1)
    else:
        beta = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = beta / Q_b
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -Q_b, Q_b - 1))
    x_ste = (x_quant - x_norm).detach() + x_norm
    return x_ste * scale


def activation_quant_ste_absmean(x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    Q_b = 2 ** (num_bits - 1)
    beta = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = beta / Q_b
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -Q_b, Q_b - 1))
    x_ste = (x_quant - x_norm).detach() + x_norm
    return x_ste * scale


def fp4_quant_ste(x: torch.Tensor, M: int = 3, b: int = 2) -> torch.Tensor:
    abs_x = x.abs()
    log2_abs = torch.log2(abs_x.clamp_min(1e-8))
    gamma = 2 ** torch.clamp(torch.floor(log2_abs) + b, min=1.0)
    denom = 2 ** (M + b)
    scale = gamma / denom
    x_quant = torch.round(x / scale)
    x_ste = (x_quant - x / scale).detach() + x / scale
    return x_ste * scale


def q_int4_ste(x: torch.Tensor) -> torch.Tensor:
    """4-bit integer quantization (absmean) for activations."""
    beta = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = beta / 7.0
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -8, 7))
    return ((x_quant - x_norm).detach() + x_norm) * scale


def q_fp4_ste(x: torch.Tensor, M: int = 2, E: int = 2) -> torch.Tensor:
    """4-bit floating-point quantization (E2M1 format)."""
    abs_x = x.abs()
    log2_abs = torch.log2(abs_x.clamp_min(1e-8))
    gamma = 2 ** torch.clamp(torch.floor(log2_abs) + (2**E - 1), min=1.0)
    denom = 2 ** (M + 2**E - 1)
    scale = gamma / denom
    x_quant = torch.round(x / scale)
    return (x_quant - x / scale).detach() + x * scale


def topk_sparsify(x: torch.Tensor, k: float = 0.5) -> torch.Tensor:
    if not x.requires_grad:
        abs_x = x.abs().float()
        threshold = torch.quantile(abs_x, 1.0 - k, dim=-1, keepdim=True)
        mask = abs_x >= threshold
        return x * mask.to(x.dtype)
    abs_x = x.abs().float()
    threshold = torch.quantile(abs_x, 1.0 - k, dim=-1, keepdim=True)
    mask = abs_x >= threshold
    return (mask.to(x.dtype) * x - x).detach() + x

def quantize_ste_absmax(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Symmetric absmax quantization with straight-through estimator."""
    Qb = 2 ** (num_bits - 1) - 1
    gamma = x.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8)
    scale = gamma / Qb
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -Qb, Qb))
    return (x_quant - x_norm).detach() + x_norm * scale


def hadamard_transform(x):
    *batch_dims, D = x.shape
    x = x.reshape(-1, D)
    pad_m = 1
    while pad_m < D:
        pad_m <<= 1
    if pad_m != D:
        x = F.pad(x, (0, pad_m - D))
    N = x.size(0)
    x = x[:, None, :]
    block = 1
    while block < pad_m:
        x = x.view(N, -1, 2, block)
        u = x[:, :, 0]
        v = x[:, :, 1]
        x = torch.stack([u + v, u - v], dim=2)
        block <<= 1
    x = x.view(N, pad_m)
    return (x[:, :D] / math.sqrt(pad_m)).reshape(*batch_dims, D)


def q_int4_v2(x):
    """4-bit INT4 для BitNet v2 (после Hadamard transform)."""
    return q_int4_ste(x)


def make_linear(
    cfg, in_features: int, out_features: int, bias: bool = False, quantize_input: bool = True
):
    if getattr(cfg, "use_bitlinear", False):
        abits = getattr(cfg, "bitnet_activation_bits", 8)
        return BitLinear(
            in_features,
            out_features,
            bias=bias,
            activation_bits=abits,
            quantize_input=quantize_input,
        )
    return nn.Linear(in_features, out_features, bias=bias)


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
        quantize_input: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.quantize_input = quantize_input
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ste_b158(self.weight).to(x.dtype)
        if self.quantize_input:
            x = activation_quant_ste(x, self.activation_bits)
        return F.linear(x, w, self.bias)


class QuantizedKVCache:
    def __init__(self, num_bits: int = 3):
        self.num_bits = num_bits
        self.k_scale = None
        self.v_scale = None
        self.k_quant = None
        self.v_quant = None

    def quantize(self, k: torch.Tensor, v: torch.Tensor):
        Q_b = 2 ** (self.num_bits - 1)
        k_max = k.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8)
        v_max = v.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8)
        self.k_scale = k_max / (Q_b - 1)
        self.v_scale = v_max / (Q_b - 1)
        self.k_quant = torch.round(torch.clamp(k / self.k_scale, -(Q_b - 1), Q_b - 1)).to(
            torch.int8
        )
        self.v_quant = torch.round(torch.clamp(v / self.v_scale, -(Q_b - 1), Q_b - 1)).to(
            torch.int8
        )

    def dequantize(self) -> tuple:
        if self.k_quant is None or self.k_scale is None:
            return None, None
        assert self.v_quant is not None and self.v_scale is not None
        k = self.k_quant.float() * self.k_scale
        v = self.v_quant.float() * self.v_scale
        return k, v

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k_quant is None:
            self.quantize(k_new, v_new)
        else:
            k_old, v_old = self.dequantize()
            k_cat = torch.cat([k_old, k_new], dim=2)
            v_cat = torch.cat([v_old, v_new], dim=2)
            self.quantize(k_cat, v_cat)

    def get(self) -> tuple:
        if self.k_quant is None:
            return None, None
        return self.dequantize()




## bulba1/model/block.py

import torch
import torch.nn as nn

from bulba1.model.bit_linear import q_int4_v2, topk_sparsify
from bulba1.model.diff_attn import DiffAttention, RMSNorm
from bulba1.model.kda import KimiDeltaAttention
from bulba1.model.mamba import MambaBlock
from bulba1.model.mhc import MHC
from bulba1.model.moe import MoELayer


class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.res_scale = nn.Parameter(torch.zeros(1))

        pattern = getattr(cfg, "alternating_pattern", None)
        if pattern is not None and layer_idx < len(pattern):
            self.is_attn_block = pattern[layer_idx] == "attn"
        else:
            attn_every = getattr(cfg, "attn_every_n_layers", 4) or 4
            self.is_attn_block = (layer_idx % attn_every) == 0

        if self.is_attn_block:
            self.norm1 = RMSNorm(cfg.d_model)
            if getattr(cfg, "use_kda", False):
                self.attn = KimiDeltaAttention(cfg)
            else:
                self.attn = DiffAttention(cfg)
            self.norm2 = RMSNorm(cfg.d_model)
            if cfg.use_moe:
                self.moe = MoELayer(cfg, layer_idx)
        else:
            self.norm1 = RMSNorm(cfg.d_model)
            if cfg.use_mamba:
                self.mamba = MambaBlock(cfg)

        self.use_mhc = getattr(cfg, "use_mhc", True)
        if self.use_mhc:
            self.mhc = MHC(cfg)

        self.use_bitnet_a48 = getattr(cfg, "use_bitnet_a48", False)
        if self.use_bitnet_a48:
            self.a48_topk = getattr(cfg, "a48_attn_topk_sparsity", 0.5)

        self.sd_prob = getattr(cfg, "stochastic_depth_prob", 0.0)

    def forward(self, x: torch.Tensor, prev_experts=None, past_kv=None):
        if self.training and self.sd_prob > 0.0 and torch.rand(1, device=x.device) < self.sd_prob:
            return x, torch.tensor(0.0, device=x.device, dtype=x.dtype), None

        h = x
        if self.use_bitnet_a48:
            h = q_int4_v2(self.norm1(h))

        if self.use_mhc:
            if self.is_attn_block:
                def attn_fn(h_in, past_kv=past_kv):
                    attn_out, new_kv, attn_z = self.attn(self.norm1(h_in), past_kv=past_kv)
                    if self.use_bitnet_a48:
                        attn_out = topk_sparsify(attn_out, self.a48_topk)
                    h_mid = h_in + attn_out
                    if self.cfg.use_moe:
                        moe_out, aux_loss = self.moe(self.norm2(h_mid), prev_experts)
                    else:
                        moe_out = torch.zeros_like(h_mid)
                        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                    h_mid = h_mid + moe_out
                    return h_mid, new_kv, aux_loss + attn_z * getattr(self.cfg, "attn_z_loss_coef", 0.0001)

                h, new_kv, total_aux = self.mhc(h, attn_fn, past_kv=past_kv)
            else:
                def mamba_fn(h_in):
                    if hasattr(self, 'mamba'):
                        mamba_out = self.mamba(self.norm1(h_in))
                    else:
                        mamba_out = torch.zeros_like(h_in)
                    if self.use_bitnet_a48:
                        mamba_out = topk_sparsify(mamba_out, self.a48_topk)
                    return h_in + mamba_out

                h = self.mhc(h, mamba_fn)
                new_kv = None
                total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        else:
            # Обычный остаточный путь (без MHC)
            if self.is_attn_block:
                attn_out, new_kv, attn_z = self.attn(self.norm1(h), past_kv=past_kv)
                if self.use_bitnet_a48:
                    attn_out = topk_sparsify(attn_out, self.a48_topk)
                h = h + attn_out

                if self.cfg.use_moe:
                    moe_out, aux_loss = self.moe(self.norm2(h), prev_experts)
                else:
                    moe_out = torch.zeros_like(h)
                    aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

                h = h + moe_out
                total_aux = aux_loss + attn_z * getattr(self.cfg, "attn_z_loss_coef", 0.0001)
            else:
                if hasattr(self, 'mamba'):
                    mamba_out = self.mamba(self.norm1(h))
                else:
                    mamba_out = torch.zeros_like(h)
                if self.use_bitnet_a48:
                    mamba_out = topk_sparsify(mamba_out, self.a48_topk)
                h = h + mamba_out
                total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                new_kv = None

        h = x + (h - x) * self.res_scale.tanh()
        return h, total_aux, new_kv




## bulba1/model/diff_attn.py

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bulba1.model.bit_linear import (
    activation_quant_ste,  # недостающий импорт
    make_linear,
    quantize_ste_absmax,
)


class QuantizedKVCache:
    """Заглушка для квантизованного кэша, если понадобится."""

    def __init__(self, num_bits: int = 3):
        self.num_bits = num_bits
        self.keys = []
        self.values = []

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self.keys.append(k)
        self.values.append(v)

    def get(self):
        if not self.keys:
            return None, None
        return torch.cat(self.keys, dim=2), torch.cat(self.values, dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def apply_yarn(self, new_max_len: int, scale: float | None = None):
        if new_max_len <= self.max_seq_len:
            return
        if scale is None:
            scale = (new_max_len / self.max_seq_len) ** 0.5
        d2 = self.dim // 2
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        ramp = torch.linspace(0, 1, d2)
        gamma = 1.0 / (1.0 + ramp * (scale - 1.0))
        scaled_freqs = freqs * gamma
        t = torch.arange(new_max_len, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, scaled_freqs)] * 2, dim=-1)
        self.max_seq_len = new_max_len
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Индексирование кастомного буфера — basedpyright ошибочно видит Module вместо Tensor
        cos = self.cos[:, :, :seq_len, : x.shape[-1]]  # pyright: ignore[reportIndexIssue]
        sin = self.sin[:, :, :seq_len, : x.shape[-1]]  # pyright: ignore[reportIndexIssue]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin


class DiffAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.lambda_init = getattr(cfg, "lambda_init", 0.8) or 0.8
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))

        if getattr(cfg, "use_mla", False):
            self.latent_dim = cfg.mla_latent_dim
            self.kv_compress = make_linear(
                cfg, cfg.d_model, self.latent_dim * cfg.n_heads * 2, bias=False
            )
            self.k_up = make_linear(cfg, self.latent_dim * cfg.n_heads, cfg.d_model, bias=False)
            self.v_up = make_linear(cfg, self.latent_dim * cfg.n_heads, cfg.d_model, bias=False)
            self.q_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        else:
            self.q_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
            self.k_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
            self.v_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)

        o_proj_quantize = not getattr(cfg, "use_bitnet_a48", False)
        self.o_proj = make_linear(
            cfg, cfg.d_model, cfg.d_model, bias=False, quantize_input=o_proj_quantize
        )

        if cfg.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        if cfg.use_per_head_gating:
            self.head_gates = nn.Parameter(torch.ones(cfg.n_heads))

        self.use_quantized_kv_cache = getattr(cfg, "use_quantized_kv_cache", True)
        self.kv_cache_bits = getattr(cfg, "kv_cache_bits", 3)

        self.rope = RoPE(
            self.head_dim,
            getattr(cfg, "max_ctx_len", 4096) or 4096,
            getattr(cfg, "rope_theta", 10000.0) or 10000.0,
        )
        self.register_buffer("lambda_val", torch.tensor(self.lambda_init))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: Any | None = None,
    ) -> tuple[torch.Tensor, Any | None, torch.Tensor]:
        B, T, _ = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)

        if self.cfg.use_mla:
            kv = self.kv_compress(x)
            k_latent, v_latent = kv.chunk(2, dim=-1)
            k = self.k_up(k_latent).view(B, T, H, D)
            v = self.v_up(v_latent).view(B, T, H, D)
        else:
            k = self.k_proj(x).view(B, T, H, D)
            v = self.v_proj(x).view(B, T, H, D)

        if self.cfg.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_quantized_kv_cache:
            k = quantize_ste_absmax(k, self.kv_cache_bits)
            v = quantize_ste_absmax(v, self.kv_cache_bits)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, T)
        k = self.rope(k, T)

        use_qkv = getattr(self.cfg, "use_quantized_kv_cache", False)
        if past_kv is not None:
            if use_qkv and isinstance(past_kv, QuantizedKVCache):
                past_kv.append(k, v)  # pyright: ignore[reportAttributeAccessIssue]
                k, v = past_kv.get()  # pyright: ignore[reportAttributeAccessIssue]
                assert k is not None and v is not None
            else:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            T_total = k.shape[2]
        else:
            T_total = T

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)

        use_sliding = getattr(self.cfg, "use_sliding_window", False) and T_total > getattr(
            self.cfg, "sliding_window_size", 512
        )
        if use_sliding:
            window = self.cfg.sliding_window_size
            sw_mask = torch.ones((T, T_total), dtype=torch.bool, device=x.device)
            for i in range(T):
                start = max(0, (T_total - T + i) - window)
                end = T_total - T + i + 1
                sw_mask[i, start:end] = False
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)
            mask = sw_mask | mask if mask is not None else sw_mask

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        z_loss = (torch.logsumexp(scores, dim=-1) ** 2).mean()

        out1 = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=(mask is None and T_total == T)
        )

        q2 = q * (0.5**0.5)
        k2 = k * (0.5**0.5)
        out2 = F.scaled_dot_product_attention(
            q2, k2, v, attn_mask=mask, dropout_p=0.0, is_causal=(mask is None and T_total == T)
        )

        lambda_1 = torch.sum(self.lambda_q1 * self.lambda_k1)
        lambda_2 = torch.sum(self.lambda_q2 * self.lambda_k2)
        lambda_buffer = self.lambda_val
        assert isinstance(lambda_buffer, torch.Tensor)
        lambda_val = torch.sigmoid(lambda_1) - torch.sigmoid(lambda_2) + lambda_buffer
        out = out1 - lambda_val * out2

        if self.cfg.use_per_head_gating:
            out = out * self.head_gates.view(1, H, 1, 1)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        if getattr(self.cfg, "use_bitnet_a48", False):
            topk_sparsity = getattr(self.cfg, "a48_attn_topk_sparsity", 0.5)
            if topk_sparsity > 0 and topk_sparsity < 1.0:
                out = activation_quant_ste(out, getattr(self.cfg, "bitnet_activation_bits", 8))
                mask_sp = self._compute_topk_mask(out, topk_sparsity)
                out = out * mask_sp

        out = self.o_proj(out)

        if use_qkv:
            cache = QuantizedKVCache(num_bits=getattr(self.cfg, "kv_cache_bits", 3))
            # quantize метод добавим позже, пока заглушка
            cache.append(k, v)  # pyright: ignore[reportAttributeAccessIssue]
            new_kv = cache
        else:
            new_kv = (k, v)

        return out, new_kv, z_loss

    def _compute_topk_mask(self, x: torch.Tensor, sparsity: float) -> torch.Tensor:
        k = max(1, int(x.size(-1) * sparsity))
        topk_vals, topk_idx = torch.topk(x.abs(), k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask




## bulba1/model/kda.py

import torch
import torch.nn as nn

from bulba1.model.bit_linear import make_linear


def _parallel_scan_affine(a, B):
    B_batch, T, H, D, _ = B.shape
    if T == 1:
        return B

    # Create clean copies detached from computation graph
    a_cum = a.detach().clone()
    B_cum = B.detach().clone()

    step = 1
    while step < T:
        # Calculate new values for the right half
        left_size = T - step
        a_left = a_cum[:, :left_size]
        B_left = B_cum[:, :left_size]
        
        # Compute new values (non-inplace)
        new_a = a_left * a_cum[:, step:]
        new_B = new_a * B_left + B_cum[:, step:]
        
        # Create new tensors for the next iteration
        a_new = torch.zeros_like(a_cum)
        B_new = torch.zeros_like(B_cum)
        a_new[:, :step] = a_cum[:, :step]
        a_new[:, step:] = new_a
        B_new[:, :step] = B_cum[:, :step]
        B_new[:, step:] = new_B
        
        a_cum = a_new
        B_cum = B_new

        step *= 2

    return B_cum


class _RoPEHelper:
    """RoPE implementaton that works across all PyTorch versions."""
    def __init__(self, theta=10000.0):
        self.theta = theta

    def __call__(self, q, k):
        B, H, T, D = q.shape
        position = torch.arange(T, device=q.device).float().view(T, 1)
        dim = torch.arange(0, D, 2).float().to(q.device)
        freqs = 1.0 / (self.theta ** (dim / D))
        angles = position * freqs
        cos = torch.cos(angles).view(1, 1, T, -1)
        sin = torch.sin(angles).view(1, 1, T, -1)

        def rotate_half(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            rotated = torch.empty_like(x)
            rotated[..., ::2] = x1 * cos - x2 * sin
            rotated[..., 1::2] = x1 * sin + x2 * cos
            return rotated

        return rotate_half(q), rotate_half(k)


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.gate_dim = getattr(cfg, "kda_gate_dim", 16)
        self.use_parallel_scan = getattr(cfg, "kda_use_parallel_scan", True)
        self.use_double_gate = getattr(cfg, "kda_double_gate", True)
        self.use_rope = getattr(cfg, "kda_use_rope", True)
        self.use_quantized_kv_cache = getattr(cfg, "use_quantized_kv_cache", True)
        self.kv_cache_bits = getattr(cfg, "kv_cache_bits", 3)

        self.q_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)

        if self.use_double_gate:
            self.gate_proj = make_linear(
                cfg, cfg.d_model, self.n_heads * self.gate_dim * 2, bias=False
            )
            self.gate_out = nn.Linear(self.gate_dim * 2, 2, bias=False)
        else:
            self.gate_proj = make_linear(
                cfg, cfg.d_model, self.n_heads * self.gate_dim, bias=False
            )
            self.gate_out = nn.Linear(self.gate_dim, 1, bias=False)

        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

    def _forward_sequential(self, q, k, v, gate_f, gate_i=None):
        B, H, T, D = q.shape
        S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        ys = []
        for t in range(T):
            k_t = k[:, :, t, :].unsqueeze(-1)
            v_t = v[:, :, t, :].unsqueeze(-2)
            f = gate_f[:, :, t].unsqueeze(-1).unsqueeze(-1)
            if gate_i is not None:
                i = gate_i[:, :, t].unsqueeze(-1).unsqueeze(-1)
            else:
                i = 1 - f
            S = f * S + i * (k_t @ v_t)
            q_t = q[:, :, t, :].unsqueeze(-1)
            y = (q_t.transpose(-2, -1) @ S).squeeze(-2)
            ys.append(y)
        return torch.stack(ys, dim=2)

    def _forward_parallel(self, q, k, v, gate_f, gate_i=None):
        B, H, T, D = q.shape
        if gate_i is None:
            g = gate_f.unsqueeze(-1).unsqueeze(-1)
            a = g
            B_mat = (1 - g) * (k.unsqueeze(-1) @ v.unsqueeze(-2))
        else:
            f = gate_f.unsqueeze(-1).unsqueeze(-1)
            i = gate_i.unsqueeze(-1).unsqueeze(-1)
            a = f
            B_mat = i * (k.unsqueeze(-1) @ v.unsqueeze(-2))

        # Защита от численного взрыва
        B_mat = torch.clamp(B_mat, min=-1e4, max=1e4)

        a = a.permute(0, 2, 1, 3, 4)
        B_mat = B_mat.permute(0, 2, 1, 3, 4)

        S = _parallel_scan_affine(a, B_mat)
        S = S.permute(0, 2, 1, 3, 4)

        q_exp = q.unsqueeze(-2)
        out = torch.matmul(q_exp, S).squeeze(-2)
        return out

    def forward(self, x, mask=None, past_kv=None):
        B, T, _ = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if self.use_quantized_kv_cache:
            from bulba1.model.bit_linear import quantize_ste_absmax
            k = quantize_ste_absmax(k, self.kv_cache_bits)
            v = quantize_ste_absmax(v, self.kv_cache_bits)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            if not hasattr(self, '_rope_helper'):
                self._rope_helper = _RoPEHelper()
            q, k = self._rope_helper(q, k)

        gate_logits = self.gate_proj(x).view(B, T, H, -1)
        if self.use_double_gate:
            gate = torch.sigmoid(self.gate_out(gate_logits))
            gate_f = gate[..., 0].transpose(1, 2)
            gate_i = gate[..., 1].transpose(1, 2)
        else:
            gate = torch.sigmoid(self.gate_out(gate_logits).squeeze(-1))
            gate_f = gate.transpose(1, 2)
            gate_i = None

        if self.use_parallel_scan and T > 1:
            out = self._forward_parallel(q, k, v, gate_f, gate_i)
        else:
            out = self._forward_sequential(q, k, v, gate_f, gate_i)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), None, torch.tensor(0.0, device=x.device)




## bulba1/model/mamba.py

import torch.nn as nn
from mamba_ssm import Mamba as Mamba3


class MambaBlock(nn.Module):
    """Блок Mamba-3 (требуется mamba-ssm >= 2.3.0)."""
    def __init__(self, cfg):
        super().__init__()
        self.mamba = Mamba3(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )
    def forward(self, x):
        return self.mamba(x)




## bulba1/model/mhc.py

import torch
import torch.nn as nn


class MHC(nn.Module):
    """
    Усиленный MHC (DeepSeek) с защитами от взрыва активаций.
    - n=4 (полноценный residual stream)
    - H_res включён (Sinkhorn-Knopp)
    - Дополнительные clamp'ы и нормализация
    - Совместим с autocast (все операции в том же dtype)
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n = getattr(cfg, "mhc_n", 4)
        self.iterations = getattr(cfg, "mhc_iterations", 10)

        # Только H_pre и H_post – H_res не создаётся, он вычисляется на лету
        self.phi_pre = nn.Linear(self.d_model, self.n, bias=False)
        self.phi_post = nn.Linear(self.d_model, self.n, bias=False)

        self.alpha_pre = nn.Parameter(torch.ones(1) * 0.01)
        self.alpha_post = nn.Parameter(torch.ones(1) * 0.01)
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))

        # Параметры для H_res (смешивание потоков)
        self.phi_res = nn.Linear(self.d_model, self.n * self.n, bias=False)
        self.alpha_res = nn.Parameter(torch.ones(1) * 0.01)
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

    def sinkhorn_knopp(self, M, iterations):
        for _ in range(iterations):
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        return M

    def forward(self, x, residual_fn, *args, past_kv=None, **kwargs):
        B, T, C = x.shape
        n = self.n

        # 1. Расширяем residual stream до n копий
        x_expanded = x.unsqueeze(2).expand(-1, -1, n, -1)   # (B,T,n,C)

        # 2. Агрегированное представление (среднее по потокам)
        x_flat = x_expanded.mean(dim=2)                      # (B,T,C)

        # 3. H_pre – агрегация потоков для входа подслоя
        h_pre_raw = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        H_pre = torch.softmax(h_pre_raw, dim=-1)             # выпуклая комбинация
        H_pre = H_pre.clamp(min=1e-4, max=1.0)               # защита от вырождения

        # 4. H_post – модуляция выхода подслоя
        h_post_raw = self.alpha_post * self.phi_post(x_flat) + self.b_post
        H_post = 0.5 * torch.tanh(h_post_raw)                # диапазон [-0.5, 0.5]
        H_post = H_post.clamp(min=-0.8, max=0.8)             # дополнительная защита

        # 5. H_res – перемешивание потоков (дважды стохастическая матрица)
        h_res_raw = self.alpha_res * self.phi_res(x_flat) + self.b_res.view(1, 1, n * n)
        h_res_raw = h_res_raw.view(B, T, n, n)
        H_res = self.sinkhorn_knopp(torch.exp(h_res_raw), self.iterations)
        H_res = H_res.clamp(min=1e-4, max=1.0)               # дважды стохастическая не должна выходить за [0,1]

        # 6. Вход для подслоя
        x_pre = (x_expanded * H_pre.unsqueeze(-1)).sum(dim=2)   # (B,T,C)

        # 7. Вызываем подслой
        if past_kv is not None:
            output = residual_fn(x_pre, past_kv=past_kv)
        else:
            output = residual_fn(x_pre)
        if isinstance(output, tuple):
            main_out = output[0]
            extras = output[1:]
        else:
            main_out = output
            extras = ()

        # 8. Применяем H_res к выходу подслоя
        main_out_expanded = main_out.unsqueeze(2).expand(-1, -1, n, -1)  # (B,T,n,C)
        main_out_mixed = torch.matmul(H_res, main_out_expanded)          # (B,T,n,n)@(B,T,n,C)->(B,T,n,C)
        main_out_mixed = main_out_mixed * H_post.unsqueeze(-1)           # модуляция
        main_out_mixed = main_out_mixed.sum(dim=2)                       # (B,T,C)

        # 9. Применяем H_res к исходному x (остаточная связь через смешивание)
        x_res = torch.matmul(H_res, x_expanded)                         # (B,T,n,C)
        x_res = x_res.sum(dim=2)                                        # (B,T,C)

        # 10. Нормализация: делим на n, т.к. суммирование n копий даёт усиление ~n
        x_res = x_res / n
        main_out_mixed = main_out_mixed / n

        # 11. Остаточная связь
        new_x = x + x_res + main_out_mixed

        # 12. Мягкое ограничение нормы (не даём вырасти больше чем на 30%)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        new_norm = new_x.norm(dim=-1, keepdim=True)
        scale = (x_norm / new_norm).clamp(max=1.3)
        new_x = new_x * scale

        if extras:
            return (new_x,) + extras
        return new_x




## bulba1/model/minichat.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from bulba1.model.bit_linear import make_linear
from bulba1.model.block import Block
from bulba1.model.diff_attn import RMSNorm
from bulba1.model.token_merging import TokenMerger
from bulba1.model.mod import MoDGate


class TiedHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        return F.linear(x, self.weight)


class MiniChat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.num_blocks = getattr(cfg, "num_unique_blocks", cfg.n_layers)
        self.repeats = getattr(cfg, "recurrent_repeats", 1)
        self.merge_every = getattr(cfg, "merge_every_n_layers", 2)
        self.use_mod = getattr(cfg, "use_mixture_of_depths", False)

        self._inference_merge_ratio = getattr(cfg, "inference_merge_ratio", 0.3)
        self.merger = None
        if self._inference_merge_ratio > 0:
            self._train_merger = TokenMerger(cfg.d_model, self._inference_merge_ratio)

        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(self.num_blocks)])
        self.norm = RMSNorm(cfg.d_model)

        if getattr(cfg, "tied_embeddings", True):
            self.lm_head = TiedHead(self.embedding.weight)
        else:
            self.lm_head = (
                make_linear(cfg, cfg.d_model, cfg.vocab_size, bias=False, quantize_input=False)
                if getattr(cfg, "bitlinear_lm_head", False)
                else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            )

        if self.use_mod:
            self.mod_gates = nn.ModuleList([
                MoDGate(cfg.d_model, getattr(cfg, "mod_capacity", 0.75))
                for _ in range(self.num_blocks)
            ])

        if cfg.use_mtp:
            self.mtp_norm = RMSNorm(cfg.d_model)
            use_bit_mtp = getattr(cfg, "bitlinear_mtp", False)
            tied = getattr(cfg, "tied_embeddings", True)
            self.mtp_projections = nn.ModuleList([
                make_linear(cfg, cfg.d_model, cfg.d_model, bias=False, quantize_input=False)
                if use_bit_mtp else nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                for _ in range(cfg.num_mtp_heads)
            ])
            if tied:
                w = self.embedding.weight
                self.mtp_heads = nn.ModuleList([TiedHead(w) for _ in range(cfg.num_mtp_heads)])
            else:
                self.mtp_heads = nn.ModuleList([
                    make_linear(cfg, cfg.d_model, cfg.vocab_size, bias=False, quantize_input=False)
                    if use_bit_mtp else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                    for _ in range(cfg.num_mtp_heads)
                ])

        self.clr_tokens = nn.Parameter(torch.randn(1, cfg.num_clr_tokens, cfg.d_model))
        self.apply(self._init_weights)

    def train(self, mode=True):
        super().train(mode)
        self.merger = None
        return self

    def eval(self):
        super().eval()
        if hasattr(self, "_train_merger") and self._train_merger is not None:
            self.merger = self._train_merger
        return self

    def _init_weights(self, module):
        init_std = getattr(self.cfg, "init_std", 0.02) or 0.02
        use_mup = getattr(self.cfg, "use_mup_init", True)
        d_model = self.cfg.d_model

        if isinstance(module, nn.Linear):
            if use_mup:
                fan_in = module.weight.size(1)
                if fan_in == d_model:
                    std = d_model ** -0.5
                else:
                    std = fan_in ** -0.5
            elif (getattr(self.cfg, "use_bitlinear", False)
                  and getattr(self.cfg, "bitnet_init_std", 0.001) > 0):
                std = getattr(self.cfg, "bitnet_init_std", 0.001)
            elif getattr(self.cfg, "depth_scaled_init", False):
                std = init_std / math.sqrt(2 * (getattr(self.cfg, "n_layers", 1) or 1))
            else:
                std = init_std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = d_model ** -0.5 if use_mup else init_std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        if self.cfg.num_clr_tokens > 0:
            clr = self.clr_tokens.expand(B, -1, -1)
            x = torch.cat([clr, x], dim=1)

        total_aux_loss = 0.0
        prev_experts = None
        use_ckpt = self.cfg.use_gradient_checkpointing
        ckpt_every = getattr(self.cfg, "checkpoint_every_n_layers", 1) or 1
        eff_hidden = self.cfg.d_model
        merge_idx = None

        for repeat in range(self.repeats):
            for i, block in enumerate(self.blocks):
                effective_layer = repeat * self.num_blocks + i

                if self.use_mod:
                    gate = self.mod_gates[i]
                    x_routed, mod_idx = gate(x)  # mod_idx: (B, n_keep)
                    if x_routed.size(1) > 0:
                        if use_ckpt and effective_layer > 0 and effective_layer % ckpt_every == 0:
                            mod_out, aux, _ = ckpt(block, x_routed, prev_experts, use_reentrant=False)
                        else:
                            mod_out, aux, _ = block(x_routed, prev_experts)
                        mod_out = mod_out.to(x.dtype)
                        # Scatter outputs back - detach indices to avoid grad issues
                        x_new = x.clone()
                        idx_expanded = mod_idx.detach().unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x_new = x_new.scatter(1, idx_expanded, mod_out)
                        x = x_new
                    else:
                        aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                elif use_ckpt and effective_layer > 0 and effective_layer % ckpt_every == 0:
                    x, aux, _ = ckpt(block, x, prev_experts, use_reentrant=False)
                else:
                    x, aux, _ = block(x, prev_experts)

                total_aux_loss += aux
                if self.cfg.use_rex and self.cfg.use_moe and hasattr(block, "moe"):
                    prev_experts = block.moe.get_expert_modules()

                if self.merger is not None and (effective_layer + 1) % self.merge_every == 0:
                    x, merge_idx = self.merger(x)

        if self.merger is not None and merge_idx is not None:
            x = self.merger.unmerge(x, merge_idx, (B, T + self.cfg.num_clr_tokens, eff_hidden))

        x = self.norm(x)
        logits = self.lm_head(x)

        if self.cfg.use_mtp:
            mtp_logits = []
            h_mtp = self.mtp_norm(x)
            for i in range(self.cfg.num_mtp_heads):
                mtp_logits.append(self.mtp_heads[i](h_mtp))
                if i < self.cfg.num_mtp_heads - 1:
                    h_mtp = F.silu(self.mtp_projections[i](h_mtp))
        else:
            mtp_logits = [None, None, None]

        return logits, mtp_logits[0], mtp_logits[1], total_aux_loss




## bulba1/model/mod.py

import torch
import torch.nn as nn


class MoDGate(nn.Module):
    def __init__(self, d_model: int, capacity: float = 0.75):
        super().__init__()
        self.capacity = capacity
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        scores = self.proj(x).squeeze(-1)
        n_keep = max(1, int(T * self.capacity))
        _, top_idx = scores.topk(n_keep, dim=-1, sorted=False)
        mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        routed = x[mask].view(B, n_keep, D)
        return routed, top_idx  # Return indices instead of mask


## bulba1/model/moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from bulba1.model.bit_linear import (
    BitLinear,
    activation_quant_ste,
    activation_quant_ste_absmean,
    ste_b158,
)


class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim, use_bitlinear=True, activation_bits=8,
                 use_relu2=False, use_absmean_down=False):
        super().__init__()
        if use_bitlinear:
            self.w1 = BitLinear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
            self.w2 = BitLinear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
            self.w3 = BitLinear(hidden_dim, d_model, bias=False, activation_bits=activation_bits)
        else:
            self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.use_relu2 = use_relu2
        self.use_absmean_down = use_absmean_down
        self.use_bitlinear = use_bitlinear
        self.activation_bits = activation_bits

    def forward(self, x):
        h = F.gelu(self.w1(x)) * self.w2(x)
        if self.use_bitlinear and self.use_absmean_down:
            h = activation_quant_ste_absmean(h, self.activation_bits)
        return self.w3(h)


class SharedExpert(nn.Module):
    def __init__(self, d_model, hidden_dim, use_bitlinear=True, activation_bits=8,
                 use_relu2=False, use_absmean_down=False):
        super().__init__()
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.w1 = (Linear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
                   if use_bitlinear else Linear(d_model, hidden_dim, bias=False))
        self.w2 = (Linear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
                   if use_bitlinear else Linear(d_model, hidden_dim, bias=False))
        self.w3 = (Linear(hidden_dim, d_model, bias=False, activation_bits=activation_bits)
                   if use_bitlinear else Linear(hidden_dim, d_model, bias=False))
        self.use_relu2 = use_relu2
        self.use_absmean_down = use_absmean_down
        self.use_bitlinear = use_bitlinear
        self.activation_bits = activation_bits

    def forward(self, x):
        if self.use_relu2:
            h = F.relu(self.w1(x)).pow(2) * self.w2(x)
        else:
            h = F.silu(self.w1(x)) * self.w2(x)
        if self.use_bitlinear and self.use_absmean_down:
            h = activation_quant_ste_absmean(h, self.activation_bits)
        return self.w3(h)


class MoELayer(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.d_model = cfg.d_model
        self.expert_hidden = cfg.expert_hidden
        self.use_bitlinear = cfg.use_bitlinear
        self.layer_idx = layer_idx
        self.use_rex = cfg.use_rex
        self.use_grouped_gemm = getattr(cfg, "use_grouped_gemm", True)
        self.num_shared_experts = getattr(cfg, "num_shared_experts", 2)
        self.router_z_loss_coef = getattr(cfg, "router_z_loss_coef", 0.001)
        self.router_entropy_coef = getattr(cfg, "router_entropy_coef", 0.001)
        self.use_expert_choice = getattr(cfg, "use_expert_choice", False)
        self.expert_choice_capacity = getattr(cfg, "expert_choice_capacity", 0)

        abits = getattr(cfg, "bitnet_activation_bits", 8)
        use_relu2 = getattr(cfg, "a48_use_relu2_glu", False)
        use_absmean_down = getattr(cfg, "use_bitnet_a48", False)

        self.shared_experts = nn.ModuleList([
            SharedExpert(cfg.d_model, cfg.expert_hidden, cfg.use_bitlinear,
                         abits, use_relu2, use_absmean_down)
            for _ in range(self.num_shared_experts)
        ])

        if self.use_grouped_gemm:
            self.grouped_experts = GroupedExperts(
                cfg.num_experts, cfg.d_model, cfg.expert_hidden,
                cfg.use_bitlinear, abits, use_relu2, use_absmean_down
            )
        else:
            self.experts = nn.ModuleList([
                Expert(cfg.d_model, cfg.expert_hidden, cfg.use_bitlinear,
                       abits, use_relu2, use_absmean_down)
                for _ in range(cfg.num_experts)
            ])

        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        if self.use_rex:
            self.reuse_weight = nn.Parameter(
                torch.ones(1) * getattr(cfg, "rex_reuse_weight", 0.3)
            )

    def _forward_grouped_single_pass(self, x_rep, expert_indices, weights):
        """Один вызов grouped_experts с уже повторёнными x и индексами."""
        out = self.grouped_experts(x_rep, expert_indices)
        return out * weights.unsqueeze(-1)

    def _forward_loop_single_pass(self, x_rep, expert_indices, weights):
        out = torch.zeros(x_rep.shape[0], x_rep.shape[1], device=x_rep.device, dtype=x_rep.dtype)
        for eid in range(self.num_experts):
            mask = expert_indices == eid
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            e_out = self.experts[eid](x_rep[idx])
            w = weights[idx].unsqueeze(-1).to(out.dtype)
            out.index_add_(0, idx, e_out.to(out.dtype) * w)
        return out

    def _get_expert_output(self, x_flat, topk_idx, topk_vals):
        """
        Вычисляет выход экспертов суммированием по k, не дублируя x_flat.
        Память оптимальна, скорость почти как у единого вызова.
        """
        output = torch.zeros(x_flat.shape[0], x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]   # (B*T,)
            weights = topk_vals[:, k]     # (B*T,)

            if self.use_grouped_gemm:
                unique_e, counts = torch.unique(expert_ids, return_counts=True)
                avg_load = counts.float().mean() if len(unique_e) > 0 else 0.0
                if self.training and avg_load < 4.0:
                    out_k = self._forward_loop_single_pass(x_flat, expert_ids, weights)
                else:
                    out_k = self._forward_grouped_single_pass(x_flat, expert_ids, weights)
            else:
                out_k = self._forward_loop_single_pass(x_flat, expert_ids, weights)

            output = output + out_k

        return output

    def forward(self, x: torch.Tensor, prev_experts=None):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        logits = self.gate(x_flat)
        router_prob = F.softmax(logits, dim=-1)

        if self.use_expert_choice:
            # Expert Choice Routing (опционально, сейчас не используется)
            capacity = self.expert_choice_capacity
            if capacity <= 0:
                capacity = int(B * T * self.top_k / self.num_experts) + 1
            topk_vals, topk_idx = torch.topk(router_prob.t(), capacity, dim=1)  # (E, capacity)
            output = torch.zeros(x_flat.shape[0], x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
            for e in range(self.num_experts):
                tokens = topk_idx[e]
                if tokens.numel() == 0:
                    continue
                vals = topk_vals[e]
                expert_in = x_flat[tokens]
                if self.use_grouped_gemm:
                    e_out = self.grouped_experts(expert_in, torch.full_like(tokens, e))
                else:
                    e_out = self.experts[e](expert_in)
                output.index_add_(0, tokens, (e_out * vals.unsqueeze(-1)).to(output.dtype))
            for se in self.shared_experts:
                output = output + se(x_flat)
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = self.router_z_loss_coef * (log_z ** 2).mean()
            total_aux = z_loss
        else:
            # Token Choice
            topk_vals, topk_idx = torch.topk(router_prob, self.top_k, dim=-1)
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

            # Один проход через экспертов
            output = self._get_expert_output(x_flat, topk_idx, topk_vals)

            # Shared experts
            for se in self.shared_experts:
                output = output + se(x_flat)

            # ReX
            if self.use_rex and prev_experts is not None and len(prev_experts) > 0:
                with torch.no_grad():
                    prev_out = torch.zeros(x_flat.shape[0], x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
                    for k in range(self.top_k):
                        k_idx = topk_idx[:, k]   # (B*T,)
                        k_vals = topk_vals[:, k]
                        for eid in range(min(self.num_experts, len(prev_experts))):
                            mask = k_idx == eid
                            if not mask.any():
                                continue
                            idx = mask.nonzero(as_tuple=True)[0]
                            p_out = prev_experts[eid](x_flat[idx])
                            prev_out.index_add_(0, idx, (p_out * k_vals[idx].unsqueeze(-1)).to(prev_out.dtype))
                rw = torch.sigmoid(self.reuse_weight)
                output = output + prev_out * rw

            # Aux losses
            aux_loss = self.num_experts * (router_prob.mean(dim=0) ** 2).sum()
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = self.router_z_loss_coef * (log_z ** 2).mean()
            entropy = -(router_prob * torch.log(router_prob + 1e-10)).sum(dim=-1).mean()
            entropy_loss = -self.router_entropy_coef * entropy
            total_aux = aux_loss + z_loss + entropy_loss

        return output.view(B, T, D), total_aux

    def get_expert_modules(self):
        if self.use_grouped_gemm:
            class ExpertWrapper:
                def __init__(self, grouped, eid):
                    self.grouped = grouped
                    self.eid = eid
                def __call__(self, x):
                    ids = torch.full((x.size(0),), self.eid, dtype=torch.long, device=x.device)
                    return self.grouped(x, ids)
            return [ExpertWrapper(self.grouped_experts, eid) for eid in range(self.num_experts)]
        return list(self.experts)




## bulba1/model/token_merging.py

import torch
import torch.nn as nn


class TokenMerger(nn.Module):
    def __init__(self, d_model: int, ratio: float = 0.3):
        super().__init__()
        self.ratio = ratio
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        n_keep = max(1, int(T * (1.0 - self.ratio)))
        if n_keep >= T:
            return x, None
        scores = self.score(x).squeeze(-1)
        _, idx = scores.topk(n_keep, dim=-1, sorted=False)
        keep = x.gather(1, idx[:, :, None].expand(-1, -1, D))
        return keep, idx

    def unmerge(self, keep: torch.Tensor, idx: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        if idx is None:
            return keep
        B, T, D = orig_shape
        out = keep.new_zeros(B, T, D)
        out.scatter_(1, idx[:, :, None].expand(-1, -1, D), keep)
        return out


## bulba1/orchestrator.py

#!/usr/bin/env python3
"""
Умный оркестратор Bulba‑1: полный цикл от загрузки данных до обучения.
Запускает скрипты download и build, затем передаёт управление cli.py.
"""

import subprocess
import sys
from pathlib import Path

class BulbaOrchestrator:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path

    def download(self):
        print("📥 Скачивание всех датасетов...")
        script = Path("scripts/download_all_datasets.py")
        if not script.exists():
            print("❌ Скрипт download_all_datasets.py не найден.")
            sys.exit(1)
        subprocess.run([sys.executable, str(script)], check=True)

    def build(self):
        print("🔤 Сборка датасета и обучение токенизатора...")
        script = Path("scripts/build_and_tokenize.py")
        if not script.exists():
            print("❌ Скрипт build_and_tokenize.py не найден.")
            sys.exit(1)
        subprocess.run([sys.executable, str(script)], check=True)

    def train(self):
        print("🚀 Запуск тренировки...")
        cmd = [sys.executable, "-m", "bulba1.cli"]
        if self.config_path:
            cmd += ["--config", self.config_path]
        subprocess.run(cmd, check=True)

    def run_full(self, skip_download=False, skip_build=False):
        if not skip_download:
            self.download()
        if not skip_build:
            self.build()
        self.train()

## bulba1/tokenizer.py

import glob
import math
import os
from collections import defaultdict

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from torch.utils.data import DataLoader, IterableDataset


class SmartTokenizer:
    """Tokenizer with automatic vocabulary size optimization.

    Automatically finds the ideal vocab_size based on:
    - Dataset characteristics (language mix, domain)
    - Tokenization efficiency (bytes per token)
    - Model size constraints
    - Compression ratio knee point detection
    """

    # Recommended vocab sizes for different model scales
    VOCAB_SIZE_GUIDELINES = {
        # (min_params, max_params): (min_vocab, max_vocab, default)
        (0, 100_000_000): (8000, 16000, 12000),
        (100_000_000, 500_000_000): (16000, 32000, 24000),
        (500_000_000, 1_500_000_000): (24000, 48000, 32000),
        (1_500_000_000, 5_000_000_000): (32000, 64000, 48000),
        (5_000_000_000, float("inf")): (48000, 100000, 64000),
    }

    # Language-specific adjustments
    LANGUAGE_MULTIPLIERS = {
        "en": 1.0,
        "code": 1.3,  # Code needs more tokens for identifiers
        "multilingual": 1.5,
        "ja": 1.4,  # Japanese/Chinese need larger vocabs
        "zh": 1.4,
        "ko": 1.3,
        "ar": 1.2,
        "ru": 1.1,
    }

    def __init__(
        self,
        vocab_size: int | None = None,  # None = auto-detect
        model_path: str = "data/tokenizer.json",
        target_params: int | None = None,  # For scaling guidelines
        auto_detect: bool = True,
        sample_size: int = 10_000_000,  # Bytes to sample for analysis
        vocab_candidates: list[int] | None = None,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.target_params = target_params
        self.auto_detect = auto_detect and vocab_size is None
        self.sample_size = sample_size
        self.tokenizer = None
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        self._analysis_results = None

        # Candidate vocab sizes for auto-detection
        if vocab_candidates is None:
            self.vocab_candidates = [8000, 12000, 16000, 24000, 32000, 48000, 64000]
        else:
            self.vocab_candidates = vocab_candidates

    def _sample_data(self, files: list[str]) -> str:
        """Sample text from training files for analysis."""
        total_bytes = 0
        samples = []

        for path in files:
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    # Read in chunks to handle large files
                    while total_bytes < self.sample_size:
                        chunk = f.read(100_000)
                        if not chunk:
                            break
                        samples.append(chunk)
                        total_bytes += len(chunk.encode("utf-8"))
            except Exception:
                continue

            if total_bytes >= self.sample_size:
                break

        return "\n".join(samples)

    def _detect_language_mix(self, text: str) -> dict[str, float]:
        """Detect language composition of the dataset."""
        # Simple heuristic based on character ranges
        total_chars = len(text)
        if total_chars == 0:
            return {"en": 1.0}

        lang_counts = defaultdict(int)

        for char in text[:100_000]:  # Sample for speed
            cp = ord(char)
            if char.isascii():
                lang_counts["en"] += 1
            elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                lang_counts["zh"] += 1
            elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
                lang_counts["ja"] += 1
            elif 0xAC00 <= cp <= 0xD7AF:
                lang_counts["ko"] += 1
            elif 0x0600 <= cp <= 0x06FF:
                lang_counts["ar"] += 1
            elif 0x0400 <= cp <= 0x04FF:
                lang_counts["ru"] += 1
            else:
                lang_counts["other"] += 1

        # Normalize
        total = sum(lang_counts.values())
        return {k: v / total for k, v in lang_counts.items()}

    def _detect_code_ratio(self, text: str) -> float:
        """Estimate what fraction of text is code."""
        # Heuristics: common code patterns
        code_indicators = [
            "def ",
            "class ",
            "function",
            "return ",
            "import ",
            "from ",
            "if __name__",
            "const ",
            "let ",
            "var ",
            "#include",
            "public class",
            "=>",
            "->",
            "{}",
            "[]",
            "===",
            "!==",
            "// ",
            "/*",
            "*/",
        ]

        sample = text[:500_000]
        indicator_count = sum(1 for ind in code_indicators if ind in sample)
        return min(indicator_count / len(code_indicators), 1.0)

    def _create_base_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=r"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+|[^\s\w]",
                    behavior="isolated",
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    def _train_candidate(self, text: str, vocab_size: int) -> tuple[Tokenizer, dict]:
        """Train a candidate tokenizer and return metrics."""
        tokenizer = self._create_base_tokenizer()

        # Write sample to temp file
        temp_path = "/tmp/smart_tokenizer_sample.txt"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(text)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            min_frequency=2,
        )

        tokenizer.train([temp_path], trainer)

        # Evaluate tokenization efficiency
        encoding = tokenizer.encode(text[:100_000])
        tokens = encoding.ids

        bytes_per_token = len(text[:100_000].encode("utf-8")) / max(len(tokens), 1)
        chars_per_token = len(text[:100_000]) / max(len(tokens), 1)

        # Calculate entropy of token distribution
        token_counts = defaultdict(int)
        for t in tokens:
            token_counts[t] += 1

        total_tokens = len(tokens)
        entropy = 0.0
        for count in token_counts.values():
            p = count / total_tokens
            entropy -= p * math.log2(p)

        # Coverage: what fraction of characters are covered (not <unk>)
        unk_count = tokens.count(tokenizer.token_to_id("<unk>"))
        coverage = 1.0 - (unk_count / max(len(tokens), 1))

        metrics = {
            "vocab_size": vocab_size,
            "bytes_per_token": bytes_per_token,
            "chars_per_token": chars_per_token,
            "entropy": entropy,
            "coverage": coverage,
            "efficiency_score": bytes_per_token * coverage,  # Higher is better
        }

        return tokenizer, metrics

    def _find_knee_point(self, metrics_list: list[dict]) -> int:
        """Find the knee point where marginal gain diminishes.

        Uses the "elbow method": find point with maximum curvature.
        """
        if len(metrics_list) < 3:
            return metrics_list[-1]["vocab_size"]

        vocab_sizes = [m["vocab_size"] for m in metrics_list]
        efficiency_scores = [m["efficiency_score"] for m in metrics_list]

        # Normalize to [0, 1]
        v_min, v_max = min(vocab_sizes), max(vocab_sizes)
        e_min, e_max = min(efficiency_scores), max(efficiency_scores)

        if e_max == e_min:
            return metrics_list[len(metrics_list) // 2]["vocab_size"]

        points = []
        for v, e in zip(vocab_sizes, efficiency_scores):
            x = (v - v_min) / (v_max - v_min)
            y = (e - e_min) / (e_max - e_min)
            points.append((x, y))

        # Find point with maximum distance from line between first and last points
        x1, y1 = points[0]
        x2, y2 = points[-1]

        max_distance = 0
        knee_idx = 0

        for i, (x0, y0) in enumerate(points):
            # Distance from point to line
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distance = numerator / max(denominator, 1e-10)

            if distance > max_distance:
                max_distance = distance
                knee_idx = i

        return metrics_list[knee_idx]["vocab_size"]

    def _apply_guidelines(
        self,
        base_vocab_size: int,
        target_params: int | None,
        language_mix: dict[str, float],
        code_ratio: float,
    ) -> int:
        """Apply model size and language guidelines."""
        adjusted = base_vocab_size

        # Apply language multipliers
        dominant_multiplier = 1.0
        for lang, ratio in language_mix.items():
            if lang in self.LANGUAGE_MULTIPLIERS and ratio > 0.1:
                # Weighted by ratio
                multiplier = 1.0 + (self.LANGUAGE_MULTIPLIERS[lang] - 1.0) * ratio
                dominant_multiplier = max(dominant_multiplier, multiplier)

        adjusted = int(adjusted * dominant_multiplier)

        # Apply code multiplier
        if code_ratio > 0.3:
            adjusted = int(adjusted * (1.0 + code_ratio * 0.3))

        # Constrain by model size guidelines
        if target_params is not None:
            for (min_p, max_p), (min_v, max_v, _default_v) in self.VOCAB_SIZE_GUIDELINES.items():
                if min_p <= target_params < max_p:
                    # If auto-detected is outside recommended range, clip it
                    adjusted = max(min_v, min(max_v, adjusted))
                    break

        # Round to nearest 1000 for clean numbers
        adjusted = round(adjusted / 1000) * 1000

        return adjusted

    def analyze(self, files: list[str]) -> dict:
        """Analyze dataset and determine optimal vocab size."""
        print("[SmartTokenizer] Starting vocabulary size analysis...")

        # Sample data
        sample_text = self._sample_data(files)
        if len(sample_text) < 1000:
            print("[SmartTokenizer] Not enough data for analysis, using default 32000")
            self.vocab_size = 32000
            return {"vocab_size": 32000, "reason": "insufficient_data"}

        print(f"[SmartTokenizer] Analyzing {len(sample_text):,} characters...")

        # Detect language and domain
        language_mix = self._detect_language_mix(sample_text)
        code_ratio = self._detect_code_ratio(sample_text)

        print(f"[SmartTokenizer] Language mix: {language_mix}")
        print(f"[SmartTokenizer] Code ratio: {code_ratio:.2%}")

        # Train candidate tokenizers
        print("[SmartTokenizer] Training candidate tokenizers...")
        metrics_list = []

        for candidate_size in self.vocab_candidates:
            print(f"  Testing vocab_size={candidate_size}...", end=" ")
            _, metrics = self._train_candidate(sample_text, candidate_size)
            metrics_list.append(metrics)
            print(f"bpt={metrics['bytes_per_token']:.2f}, coverage={metrics['coverage']:.2%}")

        # Find knee point
        knee_vocab = self._find_knee_point(metrics_list)
        print(f"[SmartTokenizer] Knee point detected at vocab_size={knee_vocab}")

        # Apply guidelines
        final_vocab = self._apply_guidelines(
            knee_vocab, self.target_params, language_mix, code_ratio
        )

        print(f"[SmartTokenizer] After guideline adjustments: vocab_size={final_vocab}")

        self._analysis_results = {
            "recommended_vocab_size": final_vocab,
            "knee_point": knee_vocab,
            "language_mix": dict(language_mix),
            "code_ratio": code_ratio,
            "candidates": metrics_list,
        }

        self.vocab_size = final_vocab
        return self._analysis_results

    def _file_line_iterator(self, files, max_lines=None):
        """Генератор строк из многих файлов, без загрузки в память."""
        count = 0
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
                        count += 1
                        if max_lines and count >= max_lines:
                            return

    def train(self, files: list[str], force_vocab_size: int | None = None):
        """Тренирует токенизатор потоково, без OOM."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if self.auto_detect and force_vocab_size is None:
            self.analyze(files)
            print(f"[SmartTokenizer] Auto-selected vocab_size={self.vocab_size}")
        elif force_vocab_size is not None:
            self.vocab_size = force_vocab_size

        print(
            f"[SmartTokenizer] Training final tokenizer with vocab_size={self.vocab_size} (streaming)..."
        )

        tokenizer = self._create_base_tokenizer()
        assert isinstance(self.vocab_size, int)
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>",
                           "<|system|>", "<|user|>", "<|thinking|>", "<|assistant|>"],
            min_frequency=2,
        )

        # Потоковая тренировка: берём первые 5 миллионов строк для скорости,
        # этого более чем достаточно для BPE и не перегружает память.
        max_examples = 5_000_000
        iterator = self._file_line_iterator(files, max_lines=max_examples)
        tokenizer.train_from_iterator(iterator, trainer)

        tokenizer.save(self.model_path)
        self.tokenizer = tokenizer
        print(f"[SmartTokenizer] Saved to {self.model_path}")

    def load(self):
        """Load tokenizer from disk."""
        if os.path.exists(self.model_path):
            self.tokenizer = Tokenizer.from_file(self.model_path)
            # Update vocab_size from loaded tokenizer
            self.vocab_size = self.tokenizer.get_vocab_size()
        return self

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        return self.vocab_size or 32000

    def get_analysis_report(self) -> str:
        """Get a human-readable analysis report."""
        if self._analysis_results is None:
            return "No analysis performed yet."

        r = self._analysis_results
        lines = [
            "=" * 60,
            "SMART TOKENIZER ANALYSIS REPORT",
            "=" * 60,
            f"Recommended vocab_size: {r['recommended_vocab_size']:,}",
            f"Raw knee point:         {r['knee_point']:,}",
            f"Code ratio:             {r['code_ratio']:.1%}",
            "",
            "Language mix:",
        ]
        for lang, ratio in r["language_mix"].items():
            lines.append(f"  {lang}: {ratio:.1%}")

        lines.extend(
            [
                "",
                "Candidate performance:",
                "-" * 60,
                f"{'Vocab':>8} {'BPT':>8} {'CPT':>8} {'Entropy':>10} {'Coverage':>10}",
            ]
        )
        for m in r["candidates"]:
            lines.append(
                f"{m['vocab_size']:>8,} {m['bytes_per_token']:>8.2f} "
                f"{m['chars_per_token']:>8.2f} {m['entropy']:>10.2f} {m['coverage']:>9.1%}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# Backwards compatibility: HFTokenizer is now an alias for SmartTokenizer with auto_detect=False
class HFTokenizer(SmartTokenizer):
    def __init__(self, vocab_size: int = 32000, model_path: str = "data/tokenizer.json"):
        # Ensure vocab_size is always an int
        if vocab_size is None:
            vocab_size = 32000
        super().__init__(
            vocab_size=vocab_size,
            model_path=model_path,
            auto_detect=False,
        )


class FastTokenizer:
    CHAT_TOKENS = ["<|system|>", "<|user|>", "<|thinking|>", "<|assistant|>"]

    def __init__(self, model_path: str = "data/tokenizer_fast.json"):
        self.model_path = model_path
        self._tok = None
        self._chat_ids: dict[str, int] = {}

    def load(self):
        from tokenizers import Tokenizer as Tkz

        self._tok = Tkz.from_file(self.model_path)

    def add_chat_tokens(self):
        assert self._tok is not None, "Tokenizer not loaded"
        existing = {self._tok.token_to_id(t) for t in self.CHAT_TOKENS if self._tok.token_to_id(t) is not None}
        missing = [t for t in self.CHAT_TOKENS if t not in existing]
        if missing:
            added = self._tok.add_tokens(missing)
            if added > 0:
                self._tok.save(self.model_path)
        self._chat_ids = {t: self._tok.token_to_id(t) for t in self.CHAT_TOKENS}
        return self._chat_ids

    @property
    def chat_ids(self) -> dict[str, int]:
        if not self._chat_ids:
            self.add_chat_tokens()
        return self._chat_ids

    def encode_chat(self, messages: list[dict]) -> tuple[list[int], list[float]]:
        ids = []
        weights = []
        cid = self.chat_ids
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            tag = f"<|{role}|>"
            tag_id = cid.get(tag)
            if tag_id is not None:
                ids.append(tag_id)
                weights.append(0.0)
            if content:
                tok_ids = self.encode(content)
                ids.extend(tok_ids)
                if role == "thinking":
                    weights.extend([2.0] * len(tok_ids))
                elif role == "assistant":
                    weights.extend([1.0] * len(tok_ids))
                else:
                    weights.extend([0.0] * len(tok_ids))
            if msg.get("thinking") and role == "assistant":
                think_ids = self.encode(msg["thinking"])
                think_tag_id = cid.get("<|thinking|>")
                if think_tag_id is not None:
                    ids.append(think_tag_id)
                    weights.append(0.0)
                ids.extend(think_ids)
                weights.extend([2.0] * len(think_ids))
                assistant_tag = cid.get("<|assistant|>")
                if assistant_tag is not None:
                    ids.append(assistant_tag)
                    weights.append(0.0)
        return ids, weights

    @property
    def vocab_size(self) -> int:
        assert self._tok is not None, "Tokenizer not loaded"
        return self._tok.get_vocab_size()

    def encode(self, text: str):
        assert self._tok is not None, "Tokenizer not loaded"
        return self._tok.encode(text).ids

    def encode_batch(self, texts):
        assert self._tok is not None, "Tokenizer not loaded"
        return [r.ids for r in self._tok.encode_batch(texts)]

    def decode(self, ids):
        assert self._tok is not None, "Tokenizer not loaded"
        return self._tok.decode(ids)


# ---------------------------------------------------------------------------
# Изменённые классы датасетов с эпохальным перемешиванием
# ---------------------------------------------------------------------------


class TextDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: SmartTokenizer,
        data_dir: str,
        seq_len: int,
        return_target: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.return_target = return_target
        self.seed = seed
        self.epoch = 0
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True))
        self.stride = max(1, seq_len // 2)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Детерминированная новая перестановка для каждой эпохи
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_files = rng.permutation(self.files).tolist()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = shuffled_files
        else:
            per_worker = len(shuffled_files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else len(shuffled_files)
            )
            files = shuffled_files[start:end]

        buffer = []
        max_buffer_len = self.seq_len * 20
        for path in files:
            with open(path, encoding="utf-8", errors="ignore") as f:
                while True:
                    data = f.read(50000)
                    if not data:
                        break
                    ids = self.tokenizer.encode(data)
                    buffer.extend(ids)
                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[: self.seq_len + 1]
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        if self.return_target:
                            target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                            yield input_ids, target_ids
                        else:
                            yield input_ids
                        buffer = buffer[self.stride :]
                    if len(buffer) > max_buffer_len:
                        buffer = buffer[-max_buffer_len:]
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    if self.return_target:
                        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                        yield input_ids, target_ids
                    else:
                        yield input_ids
                    buffer = buffer[self.stride :]


class BinaryDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        return_target: bool = True,
        use_mmap: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.return_target = return_target
        self.seed = seed
        self.epoch = 0
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.bin"), recursive=True))
        self.stride = max(1, seq_len // 2)
        self.use_mmap = use_mmap

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_files = rng.permutation(self.files).tolist()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = shuffled_files
        else:
            per_worker = len(shuffled_files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else len(shuffled_files)
            )
            files = shuffled_files[start:end]

        for path in files:
            if self.use_mmap:
                arr = np.memmap(path, dtype=np.int32, mode="r")
            else:
                arr = np.fromfile(path, dtype=np.int32)
            idx = 0
            while idx + self.seq_len + 1 <= len(arr):
                chunk = arr[idx : idx + self.seq_len + 1]
                input_ids = torch.from_numpy(chunk[:-1].copy()).long()
                if self.return_target:
                    target_ids = torch.from_numpy(chunk[1:].copy()).long()
                    yield input_ids, target_ids
                else:
                    yield input_ids
                idx += self.stride


def create_dataloader(
    tokenizer,  # no typing to accept both FastTokenizer and HFTokenizer
    data_dir: str,
    batch_size: int,
    seq_len: int,
    num_workers: int = 0,
    shuffle: bool = True,
    return_target: bool = True,
    prefetch_factor: int = 2,
):
    bin_files = glob.glob(os.path.join(data_dir, "**/*.bin"), recursive=True)
    if bin_files:
        dataset = BinaryDataset(data_dir, seq_len, return_target=return_target)
    else:
        dataset = TextDataset(tokenizer, data_dir, seq_len, return_target=return_target)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "drop_last": True,
    }
    # pyright falsely complains about DataLoader **kwargs types; ignore
    loader = DataLoader(**loader_kwargs)  # pyright: ignore[reportArgumentType]
    return loader




## bulba1/training/checkpoint.py

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", keep_top_k: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoints = []

    def _extra_paths(self, path: str):
        return {
            "opt": path.replace(".safetensors", "_optimizer.pt"),
            "ema": path.replace(".safetensors", "_ema.pt"),
        }

    def save(
        self, model: nn.Module, optimizer, step: int, loss: float, config: dict | None = None, ema=None
    ):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.safetensors")
        state_dict = model.state_dict()
        save_file(state_dict, path)

        extras = self._extra_paths(path)
        torch.save(optimizer.state_dict(), extras["opt"])
        if ema is not None:
            torch.save(ema.state_dict(), extras["ema"])

        meta_path = path.replace(".safetensors", ".json")
        metadata = {"step": step, "loss": loss}
        if config:
            metadata["config"] = config
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        self.checkpoints.append((step, loss, path))

        best_path = os.path.join(self.checkpoint_dir, "best.safetensors")
        if loss <= (self.checkpoints[0][1] if self.checkpoints else float("inf")):
            save_file(state_dict, best_path)
            best_extras = self._extra_paths(best_path)
            torch.save(optimizer.state_dict(), best_extras["opt"])
            if ema is not None:
                torch.save(ema.state_dict(), best_extras["ema"])

        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.keep_top_k:
            old = self.checkpoints.pop()
            for suffix in [".safetensors", ".json", "_optimizer.pt", "_ema.pt"]:
                f = old[2].replace(".safetensors", suffix)
                if os.path.exists(f):
                    os.remove(f)

        return loss <= min(x[1] for x in self.checkpoints)

    def find_latest(self) -> str:
        import glob

        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.json")
        files = glob.glob(pattern)
        if not files:
            return ""
        files.sort(key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
        latest = files[-1]
        return latest.replace(".json", ".safetensors")

    def load(self, model: nn.Module, path: str, optimizer=None, ema=None):
        if path == "best":
            path = os.path.join(self.checkpoint_dir, "best.safetensors")
        elif path == "latest" or not path:
            path = self.find_latest()
        elif isinstance(path, int) or (isinstance(path, str) and path.isdigit()):
            step = int(path)
            path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.safetensors")
        if not path or not os.path.exists(path):
            return None
        state_dict = load_file(path)
        model.load_state_dict(state_dict, strict=False)

        extras = self._extra_paths(path)
        if optimizer is not None and os.path.exists(extras["opt"]):
            optimizer.load_state_dict(torch.load(extras["opt"], weights_only=False))
        if ema is not None and os.path.exists(extras["ema"]):
            ema.load_state_dict(torch.load(extras["ema"], weights_only=False))

        meta_path = path.replace(".safetensors", ".json")
        step = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            step = meta.get("step", 0)
        return step




## bulba1/training/chunked_ce.py

import torch


def chunked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 8192, ignore_index: int = -100
):
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets != ignore_index

    global_max = torch.full(
        (flat_logits.size(0),), -float("inf"), device=logits.device, dtype=logits.dtype
    )
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        local_max = flat_logits[:, start:end].max(dim=-1).values
        global_max = torch.maximum(global_max, local_max)

    exp_sum = torch.zeros_like(global_max)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        exp_sum += torch.exp(flat_logits[:, start:end] - global_max.unsqueeze(-1)).sum(dim=-1)

    log_denom = global_max + torch.log(exp_sum)

    correct_logits = torch.zeros_like(global_max)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_mask = valid_mask & (flat_targets >= start) & (flat_targets < end)
        if chunk_mask.any():
            chunk_targets = flat_targets[chunk_mask] - start
            correct_logits[chunk_mask] = (
                flat_logits[chunk_mask, start:end].gather(1, chunk_targets.unsqueeze(1)).squeeze(1)
            )

    nll = log_denom - correct_logits
    nll = nll[valid_mask]
    return nll.mean()


## bulba1/training/ema.py

from typing import Any

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup_cache: dict[str, torch.Tensor] = {}
        self._param_names = []

        # Однократно получаем все параметры и их имена
        self._param_dict = dict(model.named_parameters())
        for name, param in self._param_dict.items():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self._param_names.append(name)

    def update(self, model: nn.Module) -> None:
        """Обновляет EMA-тени, используя кэшированный dict."""
        params = []
        shadows = []
        for name in self._param_names:
            param = self._param_dict[name]  # dict уже создан, O(1) доступ
            shadows.append(self.shadow[name])
            params.append(param.data)

        if not params:
            return

        torch._foreach_mul_(shadows, self.decay)
        torch._foreach_add_(shadows, params, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        for name in self._param_names:
            param = self._param_dict[name]
            if name not in self._backup_cache:
                self._backup_cache[name] = torch.empty_like(param.data)
            self._backup_cache[name].copy_(param.data)
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name in self._param_names:
            param = self._param_dict[name]
            if name in self._backup_cache:
                param.data.copy_(self._backup_cache[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]




## bulba1/training/engine.py

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




## bulba1/training/eval.py

import torch
import torch.nn.functional as F
from typing import Optional, List


@torch.no_grad()
def compute_perplexity(model, data_loader, device, max_batches: int = None) -> float:
    if max_batches is None:
        max_batches = getattr(model.cfg, "eval_max_batches", 10)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    clr = model.cfg.num_clr_tokens

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            input_ids, targets = batch[0].to(device), batch[1].to(device)
        else:
            input_ids = batch.to(device)
            targets = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
        dev_type = getattr(device, "type", str(device).split(":")[0])
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=model.cfg.use_f16):
            logits, _, _, aux = model(input_ids)

        T_eff = min(input_ids.size(1), logits.size(1) - clr)
        text_logits = logits[:, clr : (clr + T_eff), :].reshape(-1, model.cfg.vocab_size)
        targets_flat = targets[:, :T_eff].reshape(-1)
        loss = F.cross_entropy(text_logits, targets_flat, reduction="sum")
        total_loss += loss.item()
        total_tokens += targets_flat.numel()

    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


@torch.no_grad()
def generate_samples(
    model, tokenizer, prompts: List[str], device, max_new_tokens: int = 50, temperature: float = 0.8
) -> List[str]:
    model.eval()
    results = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        text = tokenizer.decode(output[0].tolist())
        results.append(text)
    return results


def run_eval(model, tokenizer, eval_loader, device, prompts: Optional[List[str]] = None):
    ppl = compute_perplexity(model, eval_loader, device)
    print(f"Perplexity: {ppl:.2f}")

    if prompts:
        samples = generate_samples(model, tokenizer, prompts, device)
        for prompt, sample in zip(prompts, samples):
            print(f"\nPrompt: {prompt}")
            print(f"Sample: {sample}")

    return {"perplexity": ppl}


## bulba1/training/monitor.py

import time
from dataclasses import dataclass

import psutil
import torch


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
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None


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

    def format_status(self, snap: SystemSnapshot | None = None) -> str:
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

    def check_critical(self, cfg) -> dict[str, bool]:
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

        use_ckpt = getattr(cfg, "use_gradient_checkpointing", False)
        if use_ckpt and hasattr(torch.utils.checkpoint, "checkpoint"):
            from torch.utils.checkpoint import checkpoint as ckpt

            def _run_forward(x):
                return model(x)

            with amp_ctx:
                out = ckpt(_run_forward, dummy_input, use_reentrant=False)
                logits, mtp1, mtp2, aux_loss = out
        else:
            with amp_ctx:
                logits, mtp1, mtp2, aux_loss = model(dummy_input)

        loss = logits.mean()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            return peak_vram - base_vram
        return 0.0

    model.train()
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
            batch_size = max(1, batch_size - 2)
            grad_accum = max(1, grad_accum + 1)
            if batch_size == old_bs:
                old_seq = seq_len
                seq_len = max(1, seq_len // 2)
                if seq_len == old_seq:
                    model.train()
                    return {
                        "success": False,
                        "measured_vram_mb": 0.0,
                        "batch_size": batch_size,
                        "grad_accum": grad_accum,
                        "attempts": attempt + 1,
                        "error": f"OOM at batch_size={old_bs}, seq_len={old_seq}, cannot reduce further",
                    }

    try:
        measured_vram = _test_once(batch_size, seq_len)
        model.train()
        return {
            "success": True,
            "measured_vram_mb": measured_vram,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "attempts": max_attempts + 1,
        }
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.train()
        return {
            "success": False,
            "measured_vram_mb": 0.0,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "attempts": max_attempts + 1,
            "error": f"OOM after {max_attempts} attempts, final batch_size={batch_size}, seq_len={seq_len}",
        }




## bulba1/training/optimizer.py

import math

import torch
from torch.optim import Optimizer


class MuonOptimizer(Optimizer):
    """Muon optimizer with momentum, per-parameter RMS scaling, and compiled Newton-Schulz."""

    def __init__(
        self, params, lr=3e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5, min_dim=2
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "min_dim": min_dim,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            min_dim = group["min_dim"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]

                if nesterov and momentum > 0:
                    buf.mul_(momentum).add_(grad)
                    g = grad.add(buf, alpha=momentum)
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    g = buf

                if g.dim() == 2 and min(g.size(0), g.size(1)) >= min_dim:
                    update = self._newton_schulz(g, ns_steps)
                    A, B = g.shape
                    scale = 0.2 * math.sqrt(max(A, B))
                    update.mul_(scale)
                else:
                    update = g

                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16() if G.device.type == "cuda" else G
        transpose = X.size(0) > X.size(1)
        if transpose:
            X = X.T
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * A @ B
        if transpose:
            X = X.T
        return X.to(G.dtype)


class CombinedOptimizer:
    """Все 2D матрицы → Muon, остальное → AdamW (без 8-битных фолбэков)."""

    def __init__(self, model, cfg):
        muon_params = []
        adamw_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Исключения: всё, что не обрабатывается Muon
            is_excluded = any(
                pattern in name for pattern in ("embed", "head", "lm_", "bias", "norm", "A_log", "D")
            )
            if p.dim() == 2 and min(p.size(0), p.size(1)) >= 2 and not is_excluded:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self.muon = (
            MuonOptimizer(
                muon_params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                momentum=cfg.muon_momentum,
                nesterov=cfg.muon_nesterov,
                ns_steps=cfg.muon_ns_steps,
                min_dim=2,
            )
            if muon_params
            else None
        )

        self.adamw = (
            torch.optim.AdamW(
                adamw_params,
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
                fused=True if torch.cuda.is_available() else False,
            )
            if adamw_params
            else None
        )

    def zero_grad(self):
        if self.muon is not None:
            self.muon.zero_grad()
        if self.adamw is not None:
            self.adamw.zero_grad()

    def step(self):
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if self.muon else None,
            "adamw": self.adamw.state_dict() if self.adamw else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon and state_dict.get("muon"):
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw and state_dict.get("adamw"):
            self.adamw.load_state_dict(state_dict["adamw"])

    @property
    def param_groups(self):
        groups = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups




## bulba1/training/stages.py

from enum import Enum
from typing import List


class TrainingStage(Enum):
    Warmstart = 0
    DensitySwitch = 1
    BinaryInvasion = 2
    Distillation = 3

    def name(self) -> str:
        names = ["Warmstart", "DensitySwitch", "BinaryInvasion", "Distillation"]
        return names[self.value]

    def lr_multiplier(self, multipliers: List[float] = None) -> float:
        if multipliers is None:
            multipliers = [1.0, 3.33, 1.0, 0.5]
        return multipliers[self.value]


def stage_for_step(step: int, total: int, boundaries: List[float] = None) -> TrainingStage:
    if total == 0:
        return TrainingStage.Warmstart
    if boundaries is None:
        boundaries = [0.25, 0.50, 0.75]
    p = step / total
    if p < boundaries[0]:
        return TrainingStage.Warmstart
    elif p < boundaries[1]:
        return TrainingStage.DensitySwitch
    elif p < boundaries[2]:
        return TrainingStage.BinaryInvasion
    else:
        return TrainingStage.Distillation


def compute_curriculum_seq_len(
    step: int,
    total: int,
    seq_lens: List[int] = None,
    boundaries: List[float] = None,
    target_seq_len: int = 1024,
) -> int:
    if seq_lens is None or len(seq_lens) == 0:
        return target_seq_len
    if boundaries is None:
        boundaries = [0.15, 0.35, 0.60]
    if len(boundaries) >= len(seq_lens):
        boundaries = boundaries[: len(seq_lens) - 1]
    p = step / total if total > 0 else 0
    for i, bound in enumerate(boundaries):
        if p < bound:
            return seq_lens[i]
    return seq_lens[-1]


## configs/auto.yaml

autonomy:
  base_lr: 0.0005
  plateau_patience: 800
  max_lr_reductions: 3
  max_warm_restarts: 2
  warmup_steps: 1250

hardware:
  target_vram_pct: 85
  vram_limit_mb: 14000

data:
  data_dir: "data/tokenized"
  max_total_steps: 50000


## configs/default.yaml

model:
  # ── Core architecture ──
  d_model: 512
  n_layers: 10
  n_heads: 8
  vocab_size: 26000

  # ── MoE ──
  num_experts: 4
  top_k: 2
  expert_hidden: 512
  use_moe: true
  use_rex: true
  rex_reuse_weight: 0.1
  num_shared_experts: 3

  # ── KDA enhancements ──
  kda_use_rope: true
  kda_double_gate: true
  use_expert_choice: false
  expert_choice_capacity: 0

  # ── Attention (DiffAttention) ──
  use_diff_attn: true
  use_mla: true
  mla_latent_dim: 32
  use_qk_norm: true
  use_per_head_gating: true
  use_value_residuals: true
  rope_theta: 10000.0
  max_ctx_len: 4096
  sliding_window_size: 512
  lambda_init: 0.8

  # ── Mamba ──
  use_mamba: true
  mamba_d_state: 128
  mamba_d_conv: 4
  mamba_expand: 2
  attn_every_n_layers: 4
  use_kda: true
  kda_use_parallel_scan: true
  kda_gate_dim: 16

  # ── MHC (DeepSeek) ──
  use_mhc: true
  mhc_n: 4
  mhc_iterations: 4

  # ── Efficiency techniques ──
  num_unique_blocks: 4
  recurrent_repeats: 3
  merge_every_n_layers: 2
  inference_merge_ratio: 0.3
  use_mixture_of_depths: true
  mod_capacity: 0.75

  # ── CLR / MTP / Skip-gram ──
  num_clr_tokens: 4
  use_mtp: true
  mtp1_warmup_steps: 1500
  mtp2_warmup_steps: 3000
  num_mtp_heads: 2
  use_skip_gram: true
  skip_gram_range: 3
  skip_gram_weight: 0.05

  # ── Initialization ──
  init_std: 0.02
  depth_scaled_init: true

  # ── Loss coefficients ──
  router_z_loss_coef: 0.001
  router_entropy_coef: 0.001
  attn_z_loss_coef: 0.0001
  loss_mtp1_weight: 0.3
  loss_mtp2_weight: 0.1
  label_smoothing: 0.05

  # ── Quantization ──
  use_bitlinear: true
  bitnet_activation_bits: 8
  use_f16: true
  use_grouped_gemm: false
  bitlinear_lm_head: false
  bitlinear_mtp: true
  bitnet_init_std: 0.001
  use_quantized_kv_cache: true
  kv_cache_bits: 3
  use_fp4: false
  use_bitnet_a48: true
  a48_attn_topk_sparsity: 0.5
  a48_use_relu2_glu: true
  a48_two_stage_training: false
  a48_stage1_steps_ratio: 0.95
  a48_stage1_bits: 8
  a48_stage2_bits: 4

  # ── Optimizer ──
  use_muon: true
  muon_nesterov: true
  muon_ns_steps: 3
  muon_min_dim: 256
  muon_momentum: 0.95
  learning_rate: 0.0005
  tied_embeddings: true
  use_mup_init: true
  use_inv_sqrt_lr: true
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  max_grad_norm: 1.0
  ema_decay: 0.999
  ema_vram_threshold: 0.45

  # ── VRAM / OOM safety ──
  vram_warn_pct: 88.0
  vram_critical_pct: 95.0
  max_batch_reductions: 6
  min_batch_size: 1
  vram_safety_factor: 0.75
  vram_overhead_mb: 3072.0
  vram_overhead_factor: 1.35

training:
  # ── Schedule ──
  seq_len: 512
  batch_size: 32
  skip_preflight: true
  grad_accum_steps: 2
  total_steps: 38035
  warmup_ratio: 0.05
  epochs: 3
  use_lr_cooldown: false
  lr_cooldown_ratio: 0.05
  use_mtp_cooldown: true
  mtp_cooldown_ratio: 0.15
  mtp_end_scale: 0.1

  # Быстрый curriculum: 1000 шагов разгона с 64 до 512 токенов
  curriculum_warmup_ratio: 0.15
  curriculum_start_seq_len: 128

  # ── Checkpointing ──
  checkpoint_every: 1000
  checkpoint_keep_top_k: 3
  checkpoint_dir: "checkpoints/run_bulba1_27m"
  log_every: 50
  eval_every: 500
  eval_max_batches: 10
  gen_every: 2000

  # ── Regularization ──
  dropout: 0.05
  label_smoothing: 0.05
  gradient_noise: 3.0e-5
  stochastic_depth_prob: 0.1
  token_dropout: 0.05

  # ── Data ──
  data_dir: "data/tokenized"
  val_data_dir: "data/tokenized"
  log_dir: "logs"
  num_workers: 0
  prefetch_factor: 4

  # ── Precision ──
  use_f16: true
  use_gradient_checkpointing: true
  compile: false

  # ── Chunked CE ──
  chunk_size: 8192
  auto_chunked_ce_threshold: 0
  ignore_index: -100

  # ── Tokenizer IDs ──
  bos_id: 1
  eos_id: 2
  pad_id: 0

  # ── Auto Training Pipeline ──
  # After main training completes, automatically run SFT and/or DPO
  auto_sft: true
  auto_sft_data: "data/sft"
  auto_sft_epochs: 3
  auto_sft_lr: 1.0e-5

  auto_dpo: true
  auto_dpo_data: "data/dpo"
  auto_dpo_epochs: 3
  auto_dpo_lr: 1.0e-6
  auto_dpo_beta: 0.1




## configs/e2e_test.yaml

model:
  # ── Core architecture ──
  d_model: 832
  n_layers: 17
  n_heads: 13
  vocab_size: 26000

  # ── MoE ──
  num_experts: 16
  top_k: 2
  expert_hidden: 832
  use_moe: true
  use_rex: true
  rex_reuse_weight: 0.3
  num_shared_experts: 2

  # ── KDA enhancements ──
  kda_use_rope: true
  kda_double_gate: true
  use_expert_choice: false
  expert_choice_capacity: 0

  # ── Attention (DiffAttention) ──
  use_diff_attn: true
  use_mla: true
  mla_latent_dim: 64
  use_qk_norm: true
  use_per_head_gating: true
  use_value_residuals: true
  rope_theta: 10000.0
  max_ctx_len: 4096
  sliding_window_size: 512
  lambda_init: 0.8

  # ── Mamba ──
  use_mamba: true
  mamba_d_state: 128
  mamba_d_conv: 4
  mamba_expand: 2
  attn_every_n_layers: 4
  use_kda: true
  kda_use_parallel_scan: true
  kda_gate_dim: 16

  # ── MHC (DeepSeek) ──
  use_mhc: true
  mhc_n: 4
  mhc_iterations: 6

  # ── CLR / MTP / Skip-gram ──
  num_clr_tokens: 4
  use_mtp: true
  mtp1_warmup_steps: 1500
  mtp2_warmup_steps: 3000
  num_mtp_heads: 2
  use_skip_gram: true
  skip_gram_range: 3
  skip_gram_weight: 0.05

  # ── Initialization ──
  init_std: 0.02
  depth_scaled_init: true

  # ── Loss coefficients ──
  router_z_loss_coef: 0.001
  router_entropy_coef: 0.001
  attn_z_loss_coef: 0.0001
  loss_mtp1_weight: 0.3
  loss_mtp2_weight: 0.1
  label_smoothing: 0.05

  # ── Quantization ──
  use_bitlinear: true
  bitnet_activation_bits: 8
  use_f16: true
  use_grouped_gemm: false
  bitlinear_lm_head: false
  bitlinear_mtp: true
  bitnet_init_std: 0.001
  use_quantized_kv_cache: true
  kv_cache_bits: 3
  use_fp4: false
  use_bitnet_a48: false
  a48_attn_topk_sparsity: 0.5
  a48_use_relu2_glu: true
  a48_two_stage_training: false
  a48_stage1_steps_ratio: 0.95
  a48_stage1_bits: 8
  a48_stage2_bits: 4

  # ── Optimizer ──
  use_muon: true
  muon_nesterov: true
  muon_ns_steps: 1
  muon_min_dim: 256
  muon_momentum: 0.95
  learning_rate: 1.5e-3
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  max_grad_norm: 1.0
  ema_decay: 0.999
  ema_vram_threshold: 0.45

  # ── VRAM / OOM safety ──
  vram_warn_pct: 88.0
  vram_critical_pct: 95.0
  max_batch_reductions: 3
  min_batch_size: 1
  vram_safety_factor: 0.75
  vram_overhead_mb: 3072.0
  vram_overhead_factor: 1.35

training:
  # ── Schedule ──
  seq_len: 512
  batch_size: 30
  grad_accum_steps: 1
  total_steps: 2
  warmup_ratio: 0.01
  use_lr_cooldown: true
  lr_cooldown_ratio: 0.05
  use_mtp_cooldown: true
  mtp_cooldown_ratio: 0.15
  mtp_end_scale: 0.1

  # Быстрый curriculum: 1000 шагов разгона с 64 до 512 токенов
  curriculum_warmup_ratio: 0.01
  curriculum_start_seq_len: 64

  # ── Checkpointing ──
  checkpoint_every: 1
  checkpoint_keep_top_k: 3
  checkpoint_dir: "checkpoints/e2e_test"
  checkpoint_every_n_layers: 4

  # ── Logging ──
  log_every: 1
  eval_every: 0
  eval_max_batches: 10

  # ── Regularization ──
  dropout: 0.05
  label_smoothing: 0.05
  gradient_noise: 3.0e-5
  stochastic_depth_prob: 0.1

  # ── Data ──
  data_dir: "data/tokenized"
  log_dir: "logs/e2e_test"
  num_workers: 4
  prefetch_factor: 2

  # ── Precision ──
  use_f16: true
  use_gradient_checkpointing: true

  # ── Generation ──
  generate_max_new_tokens: 30
  generate_top_k: 50
  generate_temperature: 0.8

  # ── Chunked CE ──
  chunk_size: 8192
  auto_chunked_ce_threshold: 50000
  ignore_index: -100

  # ── Tokenizer IDs ──
  bos_id: 1
  eos_id: 2
  pad_id: 0


## configs/smoke_test.yaml

model:
  d_model: 256
  n_layers: 4
  n_heads: 4
  vocab_size: 26000
  num_experts: 4
  top_k: 2
  expert_hidden: 256
  use_moe: true
  use_rex: true
  rex_reuse_weight: 0.3
  num_shared_experts: 1
  kda_use_rope: true
  kda_double_gate: false
  use_expert_choice: false
  expert_choice_capacity: 0
  use_diff_attn: true
  use_mla: false
  mla_latent_dim: 64
  use_qk_norm: true
  use_per_head_gating: false
  use_value_residuals: false
  rope_theta: 10000.0
  max_ctx_len: 4096
  sliding_window_size: 512
  lambda_init: 0.8
  use_mamba: true
  mamba_d_state: 64
  mamba_d_conv: 4
  mamba_expand: 2
  attn_every_n_layers: 2
  use_kda: true
  kda_use_parallel_scan: false
  kda_gate_dim: 8
  use_mhc: false
  mhc_n: 2
  mhc_iterations: 3
  num_clr_tokens: 4
  use_mtp: false
  mtp1_warmup_steps: 1500
  mtp2_warmup_steps: 3000
  num_mtp_heads: 2
  use_skip_gram: false
  skip_gram_range: 3
  skip_gram_weight: 0.05
  init_std: 0.02
  depth_scaled_init: true
  router_z_loss_coef: 0.001
  router_entropy_coef: 0.001
  attn_z_loss_coef: 0.0001
  loss_mtp1_weight: 0.3
  loss_mtp2_weight: 0.1
  label_smoothing: 0.05
  use_bitlinear: false
  bitnet_activation_bits: 8
  use_f16: true
  use_grouped_gemm: false
  bitlinear_lm_head: false
  bitlinear_mtp: false
  bitnet_init_std: 0.001
  use_quantized_kv_cache: false
  kv_cache_bits: 3
  use_fp4: false
  use_bitnet_a48: false
  a48_attn_topk_sparsity: 0.5
  a48_use_relu2_glu: true
  a48_two_stage_training: false
  a48_stage1_steps_ratio: 0.95
  a48_stage1_bits: 8
  a48_stage2_bits: 4
  use_muon: false
  muon_nesterov: true
  muon_ns_steps: 1
  muon_min_dim: 256
  muon_momentum: 0.95
  learning_rate: 1.5e-3
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  max_grad_norm: 1.0
  ema_decay: 0.999
  ema_vram_threshold: 0.45
  vram_warn_pct: 88.0
  vram_critical_pct: 95.0
  max_batch_reductions: 3
  min_batch_size: 1
  vram_safety_factor: 0.75
  vram_overhead_mb: 3072.0
  vram_overhead_factor: 1.35

training:
  seq_len: 128
  batch_size: 2
  grad_accum_steps: 1
  total_steps: 3
  warmup_ratio: 0.01
  use_lr_cooldown: false
  lr_cooldown_ratio: 0.05
  use_mtp_cooldown: false
  mtp_cooldown_ratio: 0.15
  mtp_end_scale: 0.1
  curriculum_warmup_ratio: 0.0
  curriculum_start_seq_len: 64
  checkpoint_every: 2
  checkpoint_keep_top_k: 2
  checkpoint_dir: "checkpoints/smoke_test"
  log_every: 1
  eval_every: 0
  eval_max_batches: 2
  dropout: 0.0
  label_smoothing: 0.05
  gradient_noise: 0.0
  stochastic_depth_prob: 0.0
  data_dir: "data/tokenized"
  log_dir: "logs/smoke_test"
  num_workers: 0
  prefetch_factor: 2
  use_f16: true
  use_gradient_checkpointing: true
  generate_max_new_tokens: 10
  generate_top_k: 50
  generate_temperature: 0.8
  chunk_size: 8192
  auto_chunked_ce_threshold: 50000
  ignore_index: -100
  bos_id: 1
  eos_id: 2
  pad_id: 0


## docs/ARCHITECTURE.md

# Bulba 1 Architecture

## Overview

Bulba 1 — autonomous LLM training platform for consumer GPUs. Hybrid architecture combining Mamba-3, Kimi Delta Attention (KDA), Mixture of Experts (MoE), and BitNet quantization. Now with **AutoPilot** for autonomous hyperparameter tuning.

## Technologies Used

| Technology | Purpose | Status |
|------------|---------|--------|
| **Mamba-3** | State Space Model for long-range modeling | ✅ Active |
| **KDA (Kimi Delta Attention)** | Delta attention with parallel scan | ✅ Active |
| **MoE (Mixture of Experts)** | Efficient computation with 4 experts | ✅ Active |
| **ReX** | Reuse previous layer experts | ✅ Active |
| **MHC** | Multi-Head Latent Clustering | ✅ Active |
| **AutoPilot** | Autonomous HP tuning | ✅ Active |
| **MoD** | Mixture of Depths (dynamic depth) | ✅ Active |
| **Token Merging** | Inference-time efficiency | ✅ Active |
| **Tied Embeddings** | Share LM head with embedding | ✅ Active |
| **BitNet** | Ternary weight quantization | ✅ Active |
| **MLA** | Multi-Latent Attention (compressed KV) | ✅ Active |
| **MTP** | Multi-Token Prediction | ✅ Active |

## Model Configuration

```
┌─────────────────────────────────────────────────────────────┐
│ Bulba 1 Default Config (v2 with Recurrent Blocks)          │
├─────────────────────────────────────────────────────────────┤
│ d_model = 512           # Embedding size                  │
│ n_layers (base) = 10   # Config value                    │
│ num_unique_blocks = 4   # Unique blocks before repeating   │
│ recurrent_repeats = 3   # Times to repeat blocks           │
│ Effective layers = 12   # 4 × 3 = 12 total layers          │
│ n_heads = 8             # Attention heads                  │
│ vocab_size = 26000      # Vocabulary                        │
│ num_experts = 4         # MoE experts (3 shared + 1 routed) │
│ Total params = 26.7M    # With MoD, Token Merging, TiedEmb │
└─────────────────────────────────────────────────────────────┘
```

## New Features (v2 from Dump)

### Recurrent Blocks
- `num_unique_blocks` - unique blocks before repeating
- `recurrent_repeats` - how many times to repeat blocks
- `merge_every_n_layers` - token merging frequency

### Mixture of Depths (MoD)
- Dynamic depth allocation via MoDGate
- `use_mixture_of_depths` - enable/disable
- `mod_capacity` - capacity ratio (default 0.75)

### Token Merging
- Inference-time token merging for efficiency
- `inference_merge_ratio` - ratio of tokens to merge
- `merge_every_n_layers` - merge frequency

### Tied Embeddings
- LM head tied to embedding weight (memory efficient)
- `tied_embeddings` - enable/disable (default True)

## Layer Arrangement

Attentional pattern based on `attn_every_n_layers` (default: 4):

| Layer Type | Layers | Components |
|------------|--------|------------|
| Attention Block | 0, 4, 8 | KDA/DiffAttn → MoE → MHC |
| Mamba Block | 1-3, 5-7, 9-11 | Mamba-3 → MHC |

## Core Components

### 1. Embedding + CLR Tokens

```python
self.embedding = nn.Embedding(vocab_size, d_model)
# Optional: learnable CLR tokens
clr_tokens = nn.Parameter(torch.randn(1, num_clr_tokens, d_model))
```

### 2. Attention (KDA / DiffAttn)

- Parallel scan for efficiency (`kda_use_parallel_scan=True`)
- Double gating (`kda_double_gate=True`)
- RoPE positional embeddings
- MLA (Multi-Latent Attention) for compressed KV
- Quantized KV cache (3-bit)

### 3. Mamba-3 (State Space Model)

```python
from mamba_ssm import Mamba as Mamba3
# Linear complexity, no KV cache
```

### 4. MoE + ReX

- Shared experts (always active) + routed experts
- Top-k routing (default top-2)
- ReX: reuse previous layer experts

### 5. MHC (Multi-Head Latent Clustering)

- DeepSeek-style latent clustering
- Residual stream mixing
- `mhc_n=4`, `mhc_iterations=4`

### 6. MTP (Multi-Token Prediction)

Predict multiple tokens ahead:

```python
for i in range(num_mtp_heads):
    mtp_logits[i] = mtp_head(silu(mtp_proj(h_mtp)))
    h_mtp = silu(mtp_proj(h_mtp))
```

### 7. BitNet Quantization

- BitLinear: ternary weights ({-1, 0, +1})
- 8-bit activations
- 3-bit KV cache

## Training Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    Training Loop                           │
├────────────────────────────────────────────────────────────┤
│ 1. Data Loading                                           │
│    ├─ Infinite loader → batches                           │
│    ├─ Tokenizer → input_ids, labels                      │
│    └─ Curriculum: dynamic sequence length                │
│                                                            │
│ 2. Forward Pass (with Gradient Checkpointing)             │
│    ├─ Embedding + CLR tokens                              │
│    ├─ Blocks (with Token Merging, MoD if enabled)         │
│    ├─ RMSNorm + LM Head (tied or untied)                  │
│    └─ MTP heads                                           │
│                                                            │
│ 3. Loss Computation                                        │
│    ├─ CrossEntropy (main)                                 │
│    ├─ MTP losses (t+1, t+2)                              │
│    └─ Router auxiliary losses                             │
│                                                            │
│ 4. Backward Pass (BF16 AMP)                               │
│                                                            │
│ 5. Optimizer Step                                         │
│    ├─ Muon (Newton-Schulz) for large layers               │
│    └─ AdamW for norms/embeddings                          │
│                                                            │
│ 6. AutoPilot (if --auto enabled)                         │
│    ├─ compute_lr(step) - dynamic LR                      │
│    └─ step(step, loss) - adjust hyperparameters           │
│                                                            │
│ 7. Checkpointing                                          │
└────────────────────────────────────────────────────────────┘
```

## AutoPilot (Autonomous Training)

Enable with `--auto` flag:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

AutoPilot features:
- **Dynamic LR**: Adjusts learning rate based on training progress
- **Plateau Detection**: Detects when loss plateaus
- **Warm Restarts**: Restarts with new LR when stuck
- **Hyperparameter Tuning**: Adjusts weight decay, gradient noise

```python
# AutoPilot modes: CALIBRATE → EXPLORE → EXPLOIT → PLATEAU → SGDR
engine = TrainingEngine(model, cfg, tokenizer, auto_mode=True)
# Uses autopilot.compute_lr(step) instead of cosine schedule
# Calls autopilot.step(step, loss) to adjust params
```

## VRAM Optimization

| Technique | VRAM Saved | Trade-off |
|-----------|------------|-----------|
| BF16 AMP | ~50% | None |
| Gradient Checkpointing | ~30% | 10-20% slower |
| MLA (latent KV) | ~40% | Slight quality loss |
| Token Merging (inference) | ~30% | Longer inference |
| Tied Embeddings | ~10% | Shares weights |
| use_mamba=False | ~1 GB | Lose SSM benefits |
| use_bitlinear=True | ~20% | Quantization noise |

## Default VRAM Usage (RTX 5060 Ti 16GB)

With new smaller config (d_model=512, n_layers=10):

| batch | seq | VRAM |
|-------|-----|------|
| 32 | 512 | ~12 GB |
| 24 | 512 | ~10 GB |
| 16 | 512 | ~7 GB |

## References

- Mamba-3: https://arxiv.org/abs/2603.15569
- BitNet: https://arxiv.org/abs/2309.05512
- DeepSeek-MoE: https://arxiv.org/abs/2401.06066
- RoPE: https://arxiv.org/abs/2104.09864
- Kimi k1.5: https://arxiv.org/abs/2501.12598
- BitNet b1.58: https://arxiv.org/abs/2402.17762

## docs/COMPONENTS.md

# Components Reference Guide

This guide covers all model and training components in Bulba 1 with quick reference for developers.

---

## Model Components

### Core Model

| File | Class | Purpose |
|------|-------|---------|
| `model/minichat.py` | `MiniChat` | Main model container |
| `model/minichat.py` | `TiedHead` | Tied LM head (shares embedding weight) |
| `model/block.py` | `Block` | Single layer block |

### New Features (v2)

| File | Class | Purpose |
|------|-------|---------|
| `model/token_merging.py` | `TokenMerger` | Inference-time token merging |
| `model/mod.py` | `MoDGate` | Mixture of Depths dynamic gating |

### Attention

| File | Class | Purpose |
|------|-------|---------|
| `model/kda.py` | `KimiDeltaAttention` | KDA with parallel scan |
| `model/diff_attn.py` | `DiffAttention` | Differential attention with MLA |
| `model/diff_attn.py` | `RMSNorm` | RMS normalization |
| `model/diff_attn.py` | `RoPE` | Rotary position embeddings |

### State Space Model

| File | Class | Purpose |
|------|-------|---------|
| `model/mamba.py` | `MambaBlock` | Mamba-3 SSM block |

### Mixture of Experts

| File | Class | Purpose |
|------|-------|---------|
| `model/moe.py` | `MoELayer` | MoE with routing, shared experts, ReX |
| `model/moe.py` | `Expert` | Individual expert |
| `model/moe.py` | `GroupedExperts` | Batched expert computation |
| `model/moe.py` | `SharedExpert` | Always-active shared expert |

### Latent Clustering

| File | Class | Purpose |
|------|-------|---------|
| `model/mhc.py` | `MHC` | Multi-Head Latent Clustering |

### Quantization

| File | Class | Purpose |
|------|-------|---------|
| `model/bit_linear.py` | `BitLinear` | Ternary weight quantization |
| `model/bit_linear.py` | `ste_b158` | Straight-through estimator |

---

## Training Components

### Core Training

| File | Class | Purpose |
|------|-------|---------|
| `training/engine.py` | `TrainingEngine` | Main training loop (+ AutoPilot) |
| `training/optimizer.py` | `CombinedOptimizer` | Muon + AdamW combined |
| `training/optimizer.py` | `MuonOptimizer` | Muon optimizer (orthogonal init) |
| `training/ema.py` | `EMA` | Exponential moving average |
| `training/checkpoint.py` | `CheckpointManager` | Save/load with rotation |

### AutoPilot (Autonomous Training)

| File | Class | Purpose |
|------|-------|---------|
| `autonomy.py` | `AutoPilot` | Autonomous HP tuning |

**Features:**
- Modes: CALIBRATE → EXPLORE → EXPLOIT → PLATEAU → SGDR
- Dynamic LR computation
- Plateau detection
- Warm restarts

### Other Training

| File | Purpose |
|------|---------|
| `training/stages.py` | Multi-stage training |
| `training/eval.py` | Evaluation |
| `training/monitor.py` | GPU/memory monitoring |
| `training/chunked_ce.py` | Memory-efficient cross entropy |

---

## Configuration & Utilities

| File | Class | Purpose |
|------|-------|---------|
| `config.py` | `ModelConfig` | Model config dataclass |
| `tokenizer.py` | `SmartTokenizer` | Auto-detecting tokenizer trainer |
| `tokenizer.py` | `FastTokenizer` | Fast inference tokenizer |
| `tokenizer.py` | `HFTokenizer` | HuggingFace tokenizer wrapper |
| `tokenizer.py` | `TextDataset` | Iterable text dataset |
| `tokenizer.py` | `BinaryDataset` | Iterable binary dataset |
| `cli.py` | - | CLI with --auto flag |
| `orchestrator.py` | - | End-to-end pipeline |

---

## Quick Decision Guide

### Enable/Disable New Features

| Feature | Config Option | When to Enable |
|---------|---------------|----------------|
| Token Merging | `inference_merge_ratio > 0` | Long context inference |
| Mixture of Depths | `use_mixture_of_depths: true` | Dynamic depth allocation |
| Tied Embeddings | `tied_embeddings: true` | Memory efficiency (default) |
| Recurrent Blocks | `recurrent_repeats > 1` | Deeper network with fewer params |
| AutoPilot | `--auto` flag | Autonomous training |

### Memory vs Speed Trade-offs

```
Highest VRAM     ←────────────────────→     Fastest Training
   │                                            │
   ▼                                            ▼
use_mod ON                           Gradient Checkpointing OFF
Gradient Checkpointing ON              Batch size larger
Batch size smaller                      Sequence longer
Sequence shorter                        use_mamba OFF
use_bitlinear ON                        Tied Embeddings OFF
Tied Embeddings ON
```

---

## New Config Options

```yaml
# Recurrent Architecture
num_unique_blocks: 10    # Unique blocks before repeating
recurrent_repeats: 1     # How many times to repeat blocks

# Token Merging (Inference)
inference_merge_ratio: 0.3  # Ratio of tokens to merge
merge_every_n_layers: 2     # Merge frequency

# Mixture of Depths
use_mixture_of_depths: false
mod_capacity: 0.75

# Tied Embeddings (Default ON)
tied_embeddings: true
```

---

## Testing Components

```bash
# Test model
pytest tests/test_model.py -v

# Quick debug
python -c "
import torch
from bulba1 import MiniChat, ModelConfig

cfg = ModelConfig(d_model=512, n_layers=10, vocab_size=26000)
model = MiniChat(cfg)
x = torch.randint(0, 26000, (2, 32))
logits, mtp1, mtp2, aux = model(x)
print(f'Output shape: {logits.shape}')
"
```

## docs/CONFIG_GUIDE.md

# Configuration Guide

This guide covers all configuration options in Bulba 1, organized by category.

---

## Core Model (Default: 512d, 10 layers)

| Option | Default | Description |
|--------|---------|-------------|
| `d_model` | 512 | Embedding dimension |
| `n_layers` | 10 | Number of transformer layers |
| `n_heads` | 8 | Number of attention heads |
| `vocab_size` | 26000 | Vocabulary size |

---

## Recurrent Architecture (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `num_unique_blocks` | 4 | Unique blocks before repeating |
| `recurrent_repeats` | 3 | How many times to repeat blocks |
| `merge_every_n_layers` | 2 | Token merging frequency |

**Note:** Effective layers = num_unique_blocks × recurrent_repeats

---

## Mixture of Depths (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mixture_of_depths` | true | Enable MoD dynamic gating |
| `mod_capacity` | 0.75 | Token capacity ratio (0-1) |

---

## Token Merging (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `inference_merge_ratio` | 0.3 | Ratio of tokens to merge at inference |
| `merge_every_n_layers` | 2 | Merge frequency |

---

## Mixture of Experts (MoE)

| Option | Default | Description |
|--------|---------|-------------|
| `use_moe` | true | Enable MoE |
| `num_experts` | 4 | Total experts |
| `top_k` | 2 | Experts per token |
| `expert_hidden` | 512 | Expert hidden size |
| `use_rex` | true | Enable ReX (reuse previous layer) |
| `rex_reuse_weight` | 0.1 | ReX blend weight |
| `num_shared_experts` | 3 | Always-active experts |
| `use_grouped_gemm` | false | Use grouped GEMM |

---

## Attention (KDA / DiffAttn)

| Option | Default | Description |
|--------|---------|-------------|
| `use_kda` | true | Use KDA (vs DiffAttn) |
| `use_diff_attn` | true | Enable DiffAttention |
| `use_mla` | true | Multi-Latent Attention |
| `mla_latent_dim` | 32 | MLA latent dimension |
| `use_qk_norm` | true | QK normalization |
| `use_per_head_gating` | true | Per-head gating |

**Position:**
| Option | Default | Description |
|--------|---------|-------------|
| `kda_use_rope` | true | RoPE in KDA |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `max_ctx_len` | 4096 | Max context length |
| `use_sliding_window` | false | Sliding window |
| `sliding_window_size` | 512 | Window size |

**KDA-specific:**
| Option | Default | Description |
|--------|---------|-------------|
| `kda_double_gate` | true | Double gating |
| `kda_use_parallel_scan` | true | Parallel scan (faster) |
| `kda_gate_dim` | 16 | Gate dimension |

---

## Mamba (State Space Model)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mamba` | true | Enable Mamba-3 |
| `mamba_d_state` | 128 | State dimension |
| `mamba_d_conv` | 4 | Convolution width |
| `mamba_expand` | 2 | Expansion factor |
| `attn_every_n_layers` | 4 | Attention frequency |

---

## MHC (Multi-Head Latent Clustering)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mhc` | true | Enable MHC |
| `mhc_n` | 4 | Number of residual streams |
| `mhc_iterations` | 4 | Sinkhorn-Knopp iterations |

---

## MTP (Multi-Token Prediction)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mtp` | true | Enable MTP |
| `num_mtp_heads` | 2 | Number of MTP heads |
| `mtp1_warmup_steps` | 1500 | Head 1 warmup |
| `mtp2_warmup_steps` | 3000 | Head 2 warmup |

---

## Embeddings (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `tied_embeddings` | true | Share LM head with embedding (saves memory) |
| `num_clr_tokens` | 4 | CLR tokens |

---

## Quantization

| Option | Default | Description |
|--------|---------|-------------|
| `use_bitlinear` | true | Enable BitNet quantization |
| `bitnet_activation_bits` | 8 | Activation bits |
| `use_f16` | true | Use FP16/BF16 |
| `use_quantized_kv_cache` | true | Quantize KV cache |
| `kv_cache_bits` | 3 | KV cache bits |

**BitNet a4.8:**
| Option | Default | Description |
|--------|---------|-------------|
| `use_bitnet_a48` | true | Enable 4-bit training |
| `a48_attn_topk_sparsity` | 0.5 | Attention sparsity |
| `a48_use_relu2_glu` | true | Use ReLU² GLU |

---

## Optimizer

| Option | Default | Description |
|--------|---------|-------------|
| `use_muon` | true | Enable Muon optimizer |
| `muon_nesterov` | true | Nesterov momentum |
| `muon_ns_steps` | 3 | Newton-Schulz steps |
| `muon_min_dim` | 256 | Min dimension for Muon |
| `muon_momentum` | 0.95 | Momentum |

**AdamW:**
| Option | Default | Description |
|--------|---------|-------------|
| `learning_rate` | 0.0005 | Learning rate |
| `weight_decay` | 0.1 | Weight decay |
| `beta1` | 0.9 | Adam beta 1 |
| `beta2` | 0.95 | Adam beta 2 |

**EMA:**
| Option | Default | Description |
|--------|---------|-------------|
| `ema_decay` | 0.999 | EMA decay rate |

---

## Training

| Option | Default | Description |
|--------|---------|-------------|
| `seq_len` | 512 | Sequence length |
| `batch_size` | 32 | Batch size |
| `grad_accum_steps` | 2 | Gradient accumulation |
| `total_steps` | 38035 | Total steps |
| `warmup_ratio` | 0.05 | LR warmup ratio |
| `use_lr_cooldown` | false | LR cooldown |
| `use_gradient_checkpointing` | true | Save memory |

**Checkpointing:**
| Option | Default | Description |
|--------|---------|-------------|
| `checkpoint_every` | 1000 | Steps between saves |
| `checkpoint_keep_top_k` | 3 | Keep top K |
| `checkpoint_dir` | "checkpoints/run_bulba1_67m" | Checkpoint path |

---

## Curriculum

| Option | Default | Description |
|--------|---------|-------------|
| `curriculum_warmup_ratio` | 0.15 | Warmup phase |
| `curriculum_start_seq_len` | 128 | Starting sequence length |

---

## Regularization

| Option | Default | Description |
|--------|---------|-------------|
| `dropout` | 0.05 | Dropout |
| `label_smoothing` | 0.05 | Label smoothing |
| `gradient_noise` | 3.0e-5 | Gradient noise |
| `stochastic_depth_prob` | 0.1 | Stochastic depth |
| `token_dropout` | 0.05 | Token dropout |

---

## VRAM Management

| Option | Default | Description |
|--------|---------|-------------|
| `vram_warn_pct` | 88.0 | Warning % |
| `vram_critical_pct` | 95.0 | Critical % |
| `max_batch_reductions` | 6 | Max reductions |

---

## Example Configs

### VRAM Constrained (RTX 3060 12GB)

```yaml
model:
  d_model: 512
  n_layers: 10
  num_experts: 4

training:
  batch_size: 16
  seq_len: 256
  use_gradient_checkpointing: true

model:
  use_mamba: false
  use_bitlinear: true
  tied_embeddings: true
```

### Maximum Quality (RTX 4090+)

```yaml
model:
  d_model: 768
  n_layers: 16
  num_experts: 8
  batch_size: 16
  seq_len: 1024

training:
  use_gradient_checkpointing: true
  compile: true
```

### With AutoPilot

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

AutoPilot handles LR scheduling, plateau detection, and warm restarts automatically.

---

## Auto Training Pipeline (NEW v2.1)

After main training completes, automatically run SFT and/or DPO fine-tuning.

| Option | Default | Description |
|--------|---------|-------------|
| `auto_sft` | false | Run SFT after main training |
| `auto_sft_data` | "data/sft" | SFT data directory |
| `auto_sft_epochs` | 3 | SFT epochs |
| `auto_sft_lr` | 1.0e-5 | SFT learning rate |
| `auto_dpo` | false | Run DPO after SFT |
| `auto_dpo_data` | "data/dpo" | DPO data directory |
| `auto_dpo_epochs` | 3 | DPO epochs |
| `auto_dpo_lr` | 1.0e-6 | DPO learning rate |
| `auto_dpo_beta` | 0.1 | DPO beta (KL penalty) |

**Example:**

```yaml
training:
  auto_sft: true
  auto_sft_data: "data/sft"
  auto_sft_epochs: 3
  auto_sft_lr: 1.0e-5

  auto_dpo: true
  auto_dpo_data: "data/dpo"
  auto_dpo_epochs: 3
  auto_dpo_lr: 1.0e-6
  auto_dpo_beta: 0.1
```

**Usage:**

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

The pipeline runs: **Main Training → SFT → DPO** automatically.

## docs/DEVELOPER_GUIDE.md

# Developer Guide

## Project Structure

```
bulba1-python/
├── bulba1/                    # Main package
│   ├── model/                 # Model architecture
│   │   ├── minichat.py        # Main model (MiniChat, TiedHead)
│   │   ├── block.py           # Layer block (Block)
│   │   ├── kda.py             # Kimi Delta Attention
│   │   ├── diff_attn.py       # Differential Attention
│   │   ├── mamba.py           # Mamba-3 SSM
│   │   ├── moe.py             # Mixture of Experts + ReX
│   │   ├── mhc.py             # Multi-Head Latent Clustering
│   │   ├── bit_linear.py      # BitNet quantization
│   │   ├── token_merging.py   # Token merging (new)
│   │   └── mod.py             # MoDGate (new)
│   │
│   ├── training/               # Training pipeline
│   │   ├── engine.py          # Main training loop + AutoPilot
│   │   ├── optimizer.py       # Muon + AdamW
│   │   ├── checkpoint.py      # Checkpoint management
│   │   ├── ema.py             # Exponential moving average
│   │   ├── stages.py          # Multi-stage training
│   │   ├── eval.py            # Evaluation
│   │   ├── monitor.py         # System monitoring
│   │   └── chunked_ce.py      # Memory-efficient CE
│   │
│   ├── autonomy.py            # AutoPilot (new v2 feature)
│   ├── config.py             # ModelConfig
│   ├── tokenizer.py          # Custom tokenizer
│   ├── orchestrator.py       # End-to-end pipeline
│   └── cli.py                # CLI interface (with --auto)
│
├── configs/
│   └── default.yaml           # Main config (smaller model: 512d, 10 layers)
│
├── tools/                     # Profiling & visualization
├── scripts/                   # Data scripts
├── docs/                     # Documentation
└── tests/                    # Test suite
```

## Development Setup

```bash
# Install dependencies
uv sync --extra cuda --extra dev

# Verify
python -c "from bulba1 import MiniChat; print('OK')"

# Run CLI
python -m bulba1.cli --help
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=bulba1 --cov-report=html
```

## New Features (v2 from Dump)

### 1. AutoPilot (Autonomous Training)

Enable with `--auto` flag:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

In code:

```python
from bulba1.training.engine import TrainingEngine

# With AutoPilot
engine = TrainingEngine(model, cfg, tokenizer, auto_mode=True)

# Without AutoPilot (default)
engine = TrainingEngine(model, cfg, tokenizer, auto_mode=False)
```

AutoPilot adjusts:
- Learning rate dynamically
- Detects plateaus
- Triggers warm restarts

### 2. Token Merging

Enable in config:

```yaml
model:
  inference_merge_ratio: 0.3  # 30% token merging at inference
  merge_every_n_layers: 2     # Merge every 2 layers
```

### 3. Mixture of Depths (MoD)

Enable in config:

```yaml
model:
  use_mixture_of_depths: true
  mod_capacity: 0.75  # Keep 75% of tokens
```

### 4. Recurrent Blocks

```yaml
model:
  num_unique_blocks: 10  # Unique blocks
  recurrent_repeats: 2  # Repeat twice = 20 effective layers
```

### 5. Tied Embeddings

```yaml
model:
  tied_embeddings: true  # Default, saves memory
```

## Adding New Features

### 1. New Model Component

```python
# model/my_module.py
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.linear(x)
```

### 2. Enable AutoPilot

The AutoPilot is already integrated in engine.py. Just use `--auto` flag:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

## Debugging

### Print Model Structure

```python
# List all layers
for name, module in model.named_modules():
    print(name)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total/1e6:.1f}M")
```

### Debug VRAM

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Debug AutoPilot

```python
# Check if AutoPilot is active
print(f"Auto mode: {engine.auto_mode}")
print(f"Autopilot: {engine.autopilot}")

# Get current state
if engine.autopilot:
    print(f"Mode: {engine.autopilot.state.mode}")
    print(f"LR: {engine.autopilot.current_lr}")
```

## Common Issues

### OOM

```python
# Solution 1: Reduce model size
d_model: 512  # from 768
n_layers: 10  # from 16

# Solution 2: Reduce batch
batch_size: 16

# Solution 3: Enable gradient checkpointing
use_gradient_checkpointing: true

# Solution 4: Disable features
use_mamba: false
use_bitlinear: false
use_moe: false
```

### Training Instability

```python
# Use AutoPilot for automatic adjustment
python -m bulba1.cli --config configs/default.yaml --auto
```

### Checkpoint Issues

```python
# Verify checkpoint
import torch
ckpt = torch.load("checkpoints/.../model.pt", map_location="cpu")
print(f"Step: {ckpt.get('step', 'N/A')}")

# Resume with AutoPilot state
python -m bulba1.cli --resume --checkpoint-dir checkpoints/... --auto
```

## Git Workflow

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes
# ...

# Commit
git add .
git commit -m "Add my feature"

# Push
git push origin feature/my-feature
```

## Code Review Checklist

- [ ] Type hints on public functions
- [ ] Docstrings on new features
- [ ] Tests pass
- [ ] VRAM usage acceptable (<14GB)
- [ ] Works with CUDA
- [ ] Config options have defaults
- [ ] Works with and without --auto

## Performance Tuning

| Optimization | Command/Config | Impact |
|--------------|----------------|--------|
| torch.compile | `--compile` | ~20% faster |
| Gradient checkpointing | `use_gradient_checkpointing: true` | -30% VRAM |
| Smaller model | `d_model: 512, n_layers: 10` | -40% VRAM |
| Tied embeddings | `tied_embeddings: true` | -10% VRAM |
| Token merging | `inference_merge_ratio: 0.3` | -30% inference VRAM |
| Disable MoE | `use_moe: false` | -2 GB VRAM |

## Resources

- PyTorch docs: https://pytorch.org/docs/
- Mamba-3: https://github.com/state-spaces/mamba
- BitNet: https://arxiv.org/abs/2309.05512

## docs/PAPERS.md

# Papers

## BitNet

Quantifying the Capabilities of LLMs across Scale and Precision
[Paper] https://arxiv.org/abs/2405.03146v2

BitNet: Scaling 1-bit Transformers for Large Language Models
[Paper] https://arxiv.org/abs/2310.11453v1 

The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
[Paper] https://arxiv.org/abs/2402.17764v1 

BitNet a4.8: 4-bit Activations for 1-bit LLMs
[Paper] https://arxiv.org/abs/2411.04965v1

Efficient Construction of Model Family through Progressive Training Using Model Expansion
[Paper] https://arxiv.org/abs/2504.00623v1 

BitNet b1.58 2B4T Technical Report
[Paper] https://arxiv.org/abs/2504.12285
[Web Demo] https://bitnet-demo.azurewebsites.net/
[HuggingFace] https://huggingface.co/microsoft/bitn...
[Code] https://github.com/microsoft/BitNet

[Additional Recs]
T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge
https://arxiv.org/abs/2407.00088v2

FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation
https://arxiv.org/abs/2407.07093v1

Matmul or No Matmul in the Era of 1-bit LLMs
https://arxiv.org/abs/2408.11939v2

1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs
https://arxiv.org/abs/2410.16144v2

Bitnet.cpp: Efficient Edge Inference for Ternary LLMs
https://arxiv.org/abs/2502.11880v1

Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models?
https://arxiv.org/abs/2502.11895v1

(NEW!) BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs
https://arxiv.org/abs/2504.18415

(NEW!) BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation
https://arxiv.org/abs/2506.07530


## DeepSeek

DeepSeek Engram
[Paper] https://arxiv.org/abs/2601.07372v1

mentioned papers
[DeepSeek-V3.2] https://arxiv.org/abs/2512.02556
[DeepSeek Sparse Attention] https://github.com/deepseek-ai/DeepSe...


## mHC

(ByteDance) Hyper Connections
[Paper] https://arxiv.org/abs/2409.19606

(DeepSeek) mHC: Manifold-Constrained Hyper-Connections
[Paper] https://arxiv.org/abs/2512.24880

## Kimi

Kimi K2.5 
[Paper] https://arxiv.org/abs/2602.02276
[Blog] https://www.kimi.com/blog/kimi-k2-5
[HuggingFace] https://huggingface.co/moonshotai/Kim...

Kimi K2
[Paper] https://arxiv.org/abs/2507.20534

Ling 2.0 Report
[Paper] https://arxiv.org/abs/2510.22115

Muon Is Scalable For Pre-training
[Paper] https://arxiv.org/abs/2502.16982

## Context Window

TTT-E2E
[Paper] https://arxiv.org/abs/2512.23675

Appeared papers
[Titans] https://arxiv.org/abs/2501.00663
[Kimi Linear] https://arxiv.org/abs/2510.26692

## Mamba

Mamba-3: Improved Sequence Modeling using State Space Principles
[Paper] https://arxiv.org/abs/2603.15569

Mamba: Linear-Time Sequence Modeling with Selective State Spaces
[Paper] https://arxiv.org/abs/2312.00752
[Code] https://github.com/state-spaces/mamba

Transformer: Attention Is All You Need
[Paper] https://arxiv.org/abs/1706.03762

Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
[Paper] https://arxiv.org/abs/2401.09417
[Code] https://github.com/hustvl/Vim

Efficiently Modeling Long Sequences with Structured State Spaces
[Paper] https://arxiv.org/pdf/2111.00396.pdf

Flash Attention
[Paper] https://arxiv.org/abs/2205.14135

Flash Attention 2
[Paper] https://arxiv.org/abs/2307.08691

VMamba: Visual State Space Model
[Paper] https://arxiv.org/abs/2401.10166
[Code] https://github.com/MzeroMiko/VMamba

MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts
[Paper] https://arxiv.org/abs/2401.04081

MambaByte: Token-free Selective State Space Model
[Paper] https://arxiv.org/abs/2401.13660

Repeat After Me: Transformers are Better than State Space Models at Copying
[Paper] https://arxiv.org/abs/2402.01032

## docs/TRAINING.md

# Training Guide

This guide covers the training pipeline in Bulba 1, from data preparation to checkpointing.

---

## Quick Start

```bash
# Full pipeline
make install
make data
make build
make train

# Or manually
uv sync
python scripts/download_all_datasets.py
python scripts/build_and_tokenize.py

# Normal training
python -m bulba1.cli --config configs/default.yaml

# With AutoPilot (autonomous HP tuning)
python -m bulba1.cli --config configs/default.yaml --auto
```

---

## Training Pipeline Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Training Pipeline                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ Download│ →  │ Build   │ →  │ Tokenize│ →  │ Train   │ │
│  │  Data   │    │ Dataset │    │         │    │         │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                            │
│  Download:    scripts/download_all_datasets.py           │
│  Build:       scripts/build_and_tokenize.py              │
│  Train:       bulba1.cli (with/without --auto)            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## AutoPilot (Autonomous Training)

**NEW in v2:** AutoPilot automatically adjusts hyperparameters during training.

### Enable AutoPilot

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

### How it works

```python
# In TrainingEngine
if auto_mode:
    self.autopilot = AutoPilot(self.cfg, log_dir=self.cfg.log_dir)

# LR computation
def compute_lr(self, step, total):
    if self.autopilot is not None:
        return self.autopilot.compute_lr(step)  # Dynamic LR
    return cosine_decay(step, total)  # Default

# Step adjustment
action = self.autopilot.step(step, self.ema_loss)
# Adjusts: LR, weight decay, gradient noise, stochastic depth
```

### AutoPilot Modes

- **CALIBRATE**: Find stable LR
- **EXPLORE**: Try different hyperparameters
- **EXPLOIT**: Stick with best found
- **PLATEAU**: Detect plateau, trigger restart
- **SGDR**: Cyclic LR

### AutoPilot State

```python
# Check state
print(f"Mode: {engine.autopilot.state.mode}")
print(f"LR: {engine.autopilot.current_lr}")
print(f"WD: {engine.autopilot.current_wd}")
```

---

## Training Loop

### Normal Mode (without AutoPilot)

```python
engine = TrainingEngine(model, cfg, tokenizer, device="cuda", auto_mode=False)

for step in range(total_steps):
    # 1. Get batch
    batch = get_batch(engine, step)
    
    # 2. Forward pass
    logits, mtp1, mtp2, aux_loss = model(batch.input_ids)
    
    # 3. Compute loss
    loss = compute_loss(logits, mtp1, mtp2, aux_loss, labels)
    
    # 4. Backward + optimizer step
    loss.backward()
    grad_norm = engine._optimizer_step(step, total_steps)
    
    # 5. LR schedule (cosine)
    lr = engine.compute_lr(step, total_steps)
    engine.optimizer.set_lr(lr)
    
    # 6. Log & checkpoint
    if step % log_every == 0:
        log_metrics(step, loss, grad_norm, lr)
```

### AutoPilot Mode

```python
engine = TrainingEngine(model, cfg, tokenizer, device="cuda", auto_mode=True)

for step in range(total_steps):
    batch = get_batch(engine, step)
    logits, mtp1, mtp2, aux_loss = model(batch.input_ids)
    loss = compute_loss(logits, mtp1, mtp2, aux_loss, labels)
    loss.backward()
    grad_norm = engine._optimizer_step(step, total_steps)
    
    # AutoPilot LR (instead of cosine)
    lr = engine.compute_lr(step, total_steps)  # Calls autopilot.compute_lr()
    engine.optimizer.set_lr(lr)
    
    # AutoPilot step adjustment
    if step % 100 == 0:
        action = engine.autopilot.step(step, engine.ema_loss)
        # Adjusts: LR, WD, gradient noise, SD
    
    # Checkpoint includes autopilot state
    if step % checkpoint_every == 0:
        save_checkpoint(step, include_autopilot=True)
```

---

## Curriculum Learning

Dynamic sequence length based on training progress:

```yaml
curriculum_warmup_ratio: 0.15
curriculum_start_seq_len: 128
```

Default curriculum:
| Progress | Sequence Length |
|----------|-----------------|
| 0-15% | 128 |
| 15-100% | Gradual ramp to 512 |

---

## Checkpoint Management

```python
# Checkpoint includes:
{
    "step": 1000,
    "model_state": {...},
    "optimizer_state": {...},
    "config": {...},
    "autopilot": {...}  # AutoPilot state (if enabled)
}
```

### Resume with AutoPilot

```bash
python -m bulba1.cli --resume --checkpoint-dir checkpoints/run_bulba1_67m --auto
```

AutoPilot state is automatically restored to continue autonomous tuning.

---

## Loss Computation

```python
# Main loss
main_loss = cross_entropy(logits, labels)

# MTP losses (if enabled)
mtp_loss = cfg.loss_mtp1_weight * cross_entropy(mtp1, labels)
mtp_loss += cfg.loss_mtp2_weight * cross_entropy(mtp2, labels)

# Router auxiliary losses
aux_loss = router_z_loss + router_entropy_loss

# Total
total = main_loss + mtp_loss + aux_loss
```

---

## VRAM Management

### Automatic Batch Reduction

```python
if vram_pct > 95:  # Critical
    batch_size //= 2
    max 6 reductions
```

**Config:**
```yaml
vram_warn_pct: 88.0
vram_critical_pct: 95.0
max_batch_reductions: 6
```

### Memory Optimization Techniques

| Technique | VRAM Saved |
|-----------|------------|
| Gradient Checkpointing | ~30% |
| BF16 AMP | ~50% |
| Tied Embeddings | ~10% |
| Token Merging (inference) | ~30% |

---

## Monitoring

### Logs

```python
# logs/bulba1.jsonl
{
    "step": 1000,
    "loss": 2.45,
    "lr": 0.0005,
    "grad_norm": 1.2,
    "vram_gb": 10.5,
    "tokens_per_sec": 50.0,
    "batch_size": 32,
    "seq_len": 512,
    "autopilot_mode": "EXPLOIT"  # If --auto enabled
}
```

### AutoPilot Logs

```python
# logs/autonomy.jsonl
{
    "step": 1000,
    "mode": "EXPLOIT",
    "lr": 0.0005,
    "wd": 0.1,
    "noise": 3e-5,
    "action": "CONTINUE"
}
```

---

## CLI Options

```bash
python -m bulba1.cli --help

# Training
--config CONFIG          # Config file (default: configs/default.yaml)
--device DEVICE          # Device (default: auto)
--compile               # Use torch.compile
--auto                  # Enable AutoPilot (NEW)
--resume                # Resume from checkpoint

# Generation
--generate              # Run generation after training
--prompt PROMPT         # Prompt for generation
--max-new-tokens N      # Max tokens to generate

# Evaluation
--eval-every N          # Eval every N steps
--eval-prompts P1 P2    # Prompts for eval
```

---

## Troubleshooting

### Loss not converging

```bash
# Use AutoPilot for automatic adjustment
python -m bulba1.cli --config configs/default.yaml --auto
```

### OOM Errors

```yaml
# Reduce batch
batch_size: 16

# Enable gradient checkpointing
use_gradient_checkpointing: true

# Reduce sequence
seq_len: 256

# Disable Mamba
use_mamba: false
```

### Training unstable

```yaml
# Enable MHC
use_mhc: true

# Enable EMA
ema_decay: 0.999

# Reduce dropout
dropout: 0.0
```

---

## Performance

With default config (512d, 10 layers, 32 batch):

| Hardware | VRAM | Speed | Tokens/s |
|----------|------|-------|----------|
| RTX 5060 Ti 16GB | ~12 GB | 1.0 st/s | 50-60 |
| RTX 4090 24GB | ~18 GB | 1.5 st/s | 80-100 |

## scripts/auto_config.py

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def count_tokens(data_dir: str) -> int:
    total = 0
    data_path = Path(data_dir)
    for ext in ("*.bin", "*.txt"):
        for path in data_path.rglob(ext):
            fstat = path.stat()
            if ext == "*.bin":
                total += fstat.st_size // 4
            else:
                total += fstat.st_size // 3
    return total


def estimate_params(
    d_model: int,
    n_layers: int,
    num_experts: int,
    num_shared_experts: int,
    expert_hidden: int,
    vocab_size: int = 26000,
    attn_every: int = 4,
    mamba_d_state: int = 128,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mhc_n: int = 4,
    num_mtp_heads: int = 2,
    num_clr_tokens: int = 4,
    mla_latent_dim: int = 32,
    n_heads: int | None = None,
) -> int:
    if n_heads is None:
        n_heads = max(6, (d_model // 64) & ~1)

    params = 0
    params += vocab_size * d_model * 2
    params += num_clr_tokens * d_model

    num_attn = max(1, n_layers // attn_every) + (1 if n_layers % attn_every > 0 else 0)
    num_mamba = n_layers - num_attn

    mamba_params = (
        d_model * mamba_expand * d_model * 2
        + d_model * mamba_expand * mamba_d_state
        + d_model * d_model * mamba_expand
        + d_model * mamba_expand * mamba_d_conv
        + d_model
    )

    qkv_dim = d_model + 2 * mla_latent_dim
    attn_params = (
        d_model * qkv_dim
        + mla_latent_dim * d_model
        + d_model * d_model
        + n_heads * 2 * d_model
        + d_model * 2
    )

    expert_params = d_model * expert_hidden * 3
    gate_params = d_model * num_experts
    moe_params = gate_params + (num_experts + num_shared_experts) * expert_params

    mhc_params = 2 * d_model * d_model + mhc_n * d_model * d_model

    norm_params = d_model * 2

    for i in range(n_layers):
        params += mhc_params
        if i % attn_every == 0:
            params += attn_params + moe_params + norm_params * 2
        else:
            params += mamba_params + norm_params

    if num_mtp_heads > 0:
        params += num_mtp_heads * d_model * d_model
        params += num_mtp_heads * d_model * vocab_size
        params += d_model

    return params


def estimate_vram(params: int) -> float:
    return params * 220 / (1024**2)


def derive_architecture(
    total_tokens: int,
    vram_mb: float,
    chinchilla_ratio: int = 20,
    attn_every: int = 4,
) -> dict:
    chinchilla_target = total_tokens // chinchilla_ratio
    target_vram_mb = vram_mb * 0.85

    candidates = []
    for d_model in [384, 448, 512, 576, 640, 704, 768]:
        for n_layers in [8, 10, 12, 14, 16]:
            for num_experts in [4, 6, 8]:
                for shared in [1, 2, 3]:
                    for expert_hidden in [d_model, d_model * 2 // 3, d_model * 3 // 2]:
                        if expert_hidden < 128 or expert_hidden > 1024:
                            continue
                        if num_experts <= 2:
                            continue

                        vocab_size = min(max(8000, d_model * 40), 26000)
                        n_heads = max(4, (d_model // 64) & ~1)

                        params = estimate_params(
                            d_model=d_model,
                            n_layers=n_layers,
                            num_experts=num_experts,
                            num_shared_experts=shared,
                            expert_hidden=expert_hidden,
                            vocab_size=vocab_size,
                            attn_every=attn_every,
                            n_heads=n_heads,
                        )

                        vram = estimate_vram(params)

                        if vram > target_vram_mb:
                            continue

                        score = _chinchilla_score(params, chinchilla_target)
                        candidates.append(
                            {
                                "d_model": d_model,
                                "n_layers": n_layers,
                                "n_heads": n_heads,
                                "vocab_size": vocab_size,
                                "num_experts": num_experts,
                                "num_shared_experts": shared,
                                "expert_hidden": expert_hidden,
                                "params": params,
                                "vram_est_mb": vram,
                                "score": score,
                            }
                        )

    if not candidates:
        candidates = [
            {
                "d_model": 384,
                "n_layers": 8,
                "n_heads": 6,
                "vocab_size": 16000,
                "num_experts": 4,
                "num_shared_experts": 2,
                "expert_hidden": 384,
                "params": estimate_params(384, 8, 4, 2, 384, 16000),
                "vram_est_mb": 0,
                "score": 0,
            }
        ]

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def _chinchilla_score(params: int, target: int) -> float:
    if params <= 0 or target <= 0:
        return 0.0
    ratio = params / target
    if ratio > 1.0:
        return 1.0 - (ratio - 1.0)
    return ratio


def write_config(arch: dict, output_path: str, total_tokens: int, vram_mb: float):
    config = {
        "model": {
            "d_model": arch["d_model"],
            "n_layers": arch["n_layers"],
            "n_heads": arch["n_heads"],
            "vocab_size": arch["vocab_size"],
            "num_experts": arch["num_experts"],
            "top_k": 2,
            "expert_hidden": arch["expert_hidden"],
            "use_moe": True,
            "use_rex": True,
            "rex_reuse_weight": 0.1,
            "num_shared_experts": arch["num_shared_experts"],
            "kda_use_rope": True,
            "kda_double_gate": True,
            "use_expert_choice": False,
            "expert_choice_capacity": 0,
            "use_diff_attn": True,
            "use_mla": True,
            "mla_latent_dim": 32,
            "use_qk_norm": True,
            "use_per_head_gating": True,
            "use_value_residuals": True,
            "rope_theta": 10000.0,
            "max_ctx_len": 4096,
            "sliding_window_size": 512,
            "lambda_init": 0.8,
            "use_mamba": True,
            "mamba_d_state": 128,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "attn_every_n_layers": 4,
            "use_kda": True,
            "kda_use_parallel_scan": True,
            "kda_gate_dim": 16,
            "use_mhc": True,
            "mhc_n": 4,
            "mhc_iterations": 4,
            "num_clr_tokens": 4,
            "use_mtp": True,
            "mtp1_warmup_steps": 1500,
            "mtp2_warmup_steps": 3000,
            "num_mtp_heads": 2,
            "use_skip_gram": True,
            "skip_gram_range": 3,
            "skip_gram_weight": 0.05,
            "init_std": 0.02,
            "depth_scaled_init": True,
            "router_z_loss_coef": 0.001,
            "router_entropy_coef": 0.001,
            "attn_z_loss_coef": 0.0001,
            "loss_mtp1_weight": 0.3,
            "loss_mtp2_weight": 0.1,
            "label_smoothing": 0.05,
            "use_bitlinear": True,
            "bitnet_activation_bits": 8,
            "use_f16": True,
            "use_grouped_gemm": False,
            "bitlinear_lm_head": False,
            "bitlinear_mtp": True,
            "bitnet_init_std": 0.001,
            "use_quantized_kv_cache": True,
            "kv_cache_bits": 3,
            "use_fp4": False,
            "use_bitnet_a48": False,
            "a48_attn_topk_sparsity": 0.5,
            "a48_use_relu2_glu": True,
            "a48_two_stage_training": False,
            "a48_stage1_steps_ratio": 0.95,
            "a48_stage1_bits": 8,
            "a48_stage2_bits": 4,
            "use_muon": True,
            "muon_nesterov": True,
            "muon_ns_steps": 3,
            "muon_min_dim": 256,
            "muon_momentum": 0.95,
            "learning_rate": 5e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-8,
            "max_grad_norm": 1.0,
            "ema_decay": 0.999,
            "ema_vram_threshold": 0.45,
            "vram_warn_pct": 88.0,
            "vram_critical_pct": 95.0,
            "max_batch_reductions": 3,
            "min_batch_size": 1,
            "vram_safety_factor": 0.75,
            "vram_overhead_mb": 3072.0,
            "vram_overhead_factor": 1.35,
        },
        "training": {
            "seq_len": 512,
            "batch_size": 32,
            "skip_preflight": False,
            "grad_accum_steps": 2,
            "total_steps": 25000,
            "warmup_ratio": 0.05,
            "use_lr_cooldown": False,
            "lr_cooldown_ratio": 0.05,
            "use_mtp_cooldown": False,
            "mtp_cooldown_ratio": 0.15,
            "mtp_end_scale": 0.1,
            "curriculum_warmup_ratio": 0.02,
            "curriculum_start_seq_len": 256,
            "checkpoint_every": 1000,
            "checkpoint_keep_top_k": 3,
            "checkpoint_dir": "checkpoints/run_auto",
            "log_every": 10,
            "eval_every": 0,
            "eval_max_batches": 10,
            "dropout": 0.05,
            "label_smoothing": 0.05,
            "gradient_noise": 3.0e-5,
            "stochastic_depth_prob": 0.1,
            "data_dir": "data/tokenized",
            "log_dir": "logs",
            "num_workers": 0,
            "prefetch_factor": 4,
            "use_f16": True,
            "use_gradient_checkpointing": True,
            "compile": False,
            "generate_max_new_tokens": 30,
            "generate_top_k": 50,
            "generate_temperature": 0.8,
            "chunk_size": 8192,
            "auto_chunked_ce_threshold": 0,
            "ignore_index": -100,
            "bos_id": 1,
            "eos_id": 2,
            "pad_id": 0,
        },
        "autonomy": {
            "enabled": True,
            "base_lr": 5e-4,
            "plateau_patience": 800,
            "max_lr_reductions": 3,
            "max_warm_restarts": 2,
            "warmup_steps": 1250,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Auto-derived config: {arch['params'] / 1_000_000:.1f}M params, "
                f"~{arch['vram_est_mb']:.0f}MB VRAM\n")
        f.write(f"# Chinchilla optimal: {total_tokens // 20:,} params from {total_tokens:,} tokens\n")
        f.write(f"# VRAM budget: {vram_mb:.0f} MB\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-derive model architecture.")
    parser.add_argument("--data", default="data/tokenized")
    parser.add_argument("--vram", type=float, default=14000)
    parser.add_argument("--output", default="configs/auto_derived.yaml")
    parser.add_argument("--chinchilla-ratio", type=int, default=20)
    args = parser.parse_args()

    total_tokens = count_tokens(args.data)
    if total_tokens < 100_000:
        print(f"WARNING: only {total_tokens} tokens found in {args.data}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Chinchilla optimal: {total_tokens // args.chinchilla_ratio:,} params")
    print(f"VRAM budget: {args.vram} MB (target {args.vram * 0.85:.0f} MB)")

    arch = derive_architecture(total_tokens, args.vram, args.chinchilla_ratio)

    print(f"\nBest fit:")
    print(f"  d_model={arch['d_model']}, n_layers={arch['n_layers']}, "
          f"n_heads={arch['n_heads']}, vocab={arch['vocab_size']}")
    print(f"  experts={arch['num_experts']}+{arch['num_shared_experts']}shared, "
          f"expert_hidden={arch['expert_hidden']}")
    print(f"  ~{arch['params'] / 1_000_000:.1f}M params, "
          f"~{arch['vram_est_mb']:.0f}MB VRAM")

    write_config(arch, args.output, total_tokens, args.vram)


if __name__ == "__main__":
    main()


## scripts/build_and_tokenize.py

#!/usr/bin/env python3
"""
ПОТОКОВАЯ СБОРКА ДАННЫХ БЕЗ OOM – 100% ГАРАНТИЯ
Читает parquet чанками (pyarrow), пишет txt, тренирует токенизатор на txt.
НЕ ИСПОЛЬЗУЕТ shuf, НЕ ЗАГРУЖАЕТ ВСЁ В ПАМЯТЬ.
Если скрипт прервётся, можно перезапустить – готовые txt пропустятся.
"""

import os, sys, glob, json, gzip, subprocess

# Установим pyarrow, если его нет
try:
    import pyarrow.parquet as pq
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    import pyarrow.parquet as pq

TMP = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data/tmp"
DATA_TRAIN = "data/train"
TOKENIZER_PATH = "data/tokenizer_fast.json"

os.makedirs(DATA_TRAIN, exist_ok=True)

def convert_parquet_to_txt(src, dst):
    import gc
    gc.disable()   # экономим время
    import pyarrow.dataset as ds
    dataset = ds.dataset(src, format="parquet")
    total = 0
    with open(dst, "w", encoding="utf-8") as f:
        # 2 миллиона строк за раз – оптимально для 16 ГБ RAM
        for batch in dataset.to_batches(batch_size=2_000_000):
            df = batch.to_pandas()
            for text in df["text"].fillna("").astype(str):
                if text:
                    f.write(text.replace('\n', '\x01') + '\n')
                    total += 1
    gc.enable()
    return total

def convert_jsonl_to_txt(src, dst):
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or obj.get("instruction") or obj.get("output") or ""
                if not text and "messages" in obj:
                    text = "\n".join(m.get("content","") for m in obj["messages"] if m.get("content"))
                if text:
                    fout.write(text.replace('\n', '\x01') + '\n')
            except:
                pass

def convert_json_to_txt(src, dst):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as fout:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    fout.write(item.replace('\n', '\x01') + '\n')
                elif isinstance(item, dict):
                    q = item.get("query", "")
                    a = item.get("response", "") or item.get("answer", "")
                    if q and a:
                        fout.write(f"Q: {q}\nA: {a}".replace('\n', '\x01') + '\n')

def convert_gz_to_txt(src, dst):
    with gzip.open(src, "rt", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    fout.write(text.replace('\n', '\x01') + '\n')
            except:
                pass

def main():
    print("📄 Конвертация сырых файлов в txt ...")
    files = sorted(glob.glob(os.path.join(TMP, '*')))
    for fp in files:
        fname = os.path.basename(fp)
        dst = os.path.join(DATA_TRAIN, fname.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(dst):
            continue
        if fname.endswith('.pq') or fname.endswith('.parquet'):
            print(f"  {fname} (parquet) -> {os.path.basename(dst)}")
            rows = convert_parquet_to_txt(fp, dst)
            print(f"    извлечено {rows} документов")
        elif fname.endswith('.jsonl'):
            print(f"  {fname} (jsonl) -> {os.path.basename(dst)}")
            convert_jsonl_to_txt(fp, dst)
        elif fname.endswith('.json'):
            print(f"  {fname} (json) -> {os.path.basename(dst)}")
            convert_json_to_txt(fp, dst)
        elif fname.endswith('.json.gz'):
            print(f"  {fname} (json.gz) -> {os.path.basename(dst)}")
            convert_gz_to_txt(fp, dst)

    print("\n🔤 Обучение токенизатора на txt-файлах (потоково, <2 ГБ RAM) ...")
    sys.path.insert(0, os.getcwd())
    from bulba1.tokenizer import SmartTokenizer
    txt_files = sorted(glob.glob(os.path.join(DATA_TRAIN, "*.txt")))
    if not txt_files:
        print("❌ Нет txt-файлов")
        sys.exit(1)

    tok = SmartTokenizer(
        vocab_size=None,
        model_path=TOKENIZER_PATH,
        target_params=150_000_000,
        auto_detect=True,
        sample_size=10_000_000,   # только 10 МБ для анализа!
    )
    tok.train(txt_files)
    print(tok.get_analysis_report())
    print(f"✅ Токенизатор сохранён: {TOKENIZER_PATH}")
    print("\n🎉 ВСЁ ГОТОВО! Запускайте обучение: make train")

if __name__ == "__main__":
    main()

## scripts/download_all_datasets.py

#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ ЗАГРУЗКА ДАННЫХ ДЛЯ BULBA 150M (datasets + прямые ссылки)
Использует библиотеку datasets для надёжной загрузки BookCorpus, ArXiv,
PhilPapers, MC4, StarCoder, CodeParrot, Claude Opus.
Прямые ссылки оставлены только для GLM‑5.1, Kimi K2.5, MetaMathQA (уже скачаны).
"""

import os, sys, requests, json
import datasets
from huggingface_hub import get_token
from tqdm import tqdm

TMP = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data/tmp"
os.makedirs(TMP, exist_ok=True)

HF_TOKEN = get_token()
if not HF_TOKEN:
    print("❌ Токен не найден. Выполните `hf auth login` или задайте HF_TOKEN.")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------- ЗАГРУЗКА DPO DATASETS ----------
def download_dpo_dataset(name, config, split, max_rows, local_name):
    """Загружает DPO датасет и сохраняет как JSONL с messages форматом."""
    dest = os.path.join(TMP, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print(f"  ⏭️ SKIP {local_name}")
        return True
    print(f"  📥 DPO: {local_name}...")
    try:
        ds = datasets.load_dataset(name, config, split=split, token=HF_TOKEN, trust_remote_code=True)
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=local_name):
                if "messages" in row:
                    f.write(json.dumps({"messages": row["messages"]}) + "\n")
                elif "prompt" in row and "chosen" in row:
                    prompt = row["prompt"]
                    chosen = row["chosen"]
                    if isinstance(chosen, list) and len(chosen) > 0:
                        if isinstance(chosen[0], dict) and "content" in chosen[0]:
                            chosen_text = chosen[0]["content"]
                        else:
                            chosen_text = chosen[0] if chosen else ""
                    else:
                        chosen_text = str(chosen) if chosen else ""
                    msgs = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": chosen_text}]
                    f.write(json.dumps({"messages": msgs}) + "\n")
                elif "conversations" in row:
                    msgs = [{"role": m["from"], "content": m["value"]} for m in row["conversations"]]
                    f.write(json.dumps({"messages": msgs}) + "\n")
        return os.path.getsize(dest) > 100_000
    except Exception as e:
        print(f"  ❌ Ошибка DPO {local_name}: {e}")
        return False


# ---------- ЗАГРУЗКА ЧЕРЕЗ DATASETS ----------
def download_dataset(name, config, split, field, max_rows, local_name):
    """Загружает датасет через `datasets` и сохраняет как JSONL."""
    dest = os.path.join(TMP, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  ⏭️ SKIP {local_name}")
        return True
    print(f"  📥 {local_name} via datasets...")
    try:
        ds = datasets.load_dataset(name, config, split=split, token=HF_TOKEN, trust_remote_code=True)
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=local_name):
                text = row.get(field, "")
                if text:
                    f.write(json.dumps({"text": text}) + "\n")
        return os.path.getsize(dest) > 1_000_000
    except Exception as e:
        print(f"  ❌ Ошибка datasets {local_name}: {e}")
        return False

# ---------- ПРЯМАЯ ЗАГРУЗКА (только для проверенных ссылок) ----------
def download_file(url, dest, desc, min_size=1_000_000):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"  ⏭️ SKIP {desc}")
        return True
    print(f"  📥 {desc}")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            if r.status_code == 404:
                print(f"  ❌ 404: {url}")
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        return os.path.getsize(dest) >= min_size
    except Exception as e:
        print(f"  ❌ {desc}: {e}")
        return False

def main():
    print("=" * 60)
    print("📥 ФИНАЛЬНАЯ ЗАГРУЗКА (datasets + прямые ссылки)")
    print("=" * 60)

    # 1. Уже скачанные FineWeb-Edu, C4-en, GLM-5.1, Kimi, MetaMathQA – пропускаем
    # 2. Докачиваем недостающее через datasets

    downloads = [
        # BookCorpus
        ("bookcorpus", "plain_text", "train", "text", 1_000_000, "bookcorpus.jsonl"),
        # ArXiv
        ("scientific_papers", "arxiv", "train", "article", 200_000, "arxiv.jsonl"),
        # PhilPapers
        ("cast42/philarchive", "default", "train", "text", 500_000, "philarchive.jsonl"),
        # MC4 (ru) – 2 шарда прямых ссылок
        ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ru.tfrecord-00059-of-04024.json.gz",
        "mc4_ru_00059.json.gz", "MC4 (ru) part 59", 300_000_000),
        ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ru.tfrecord-00060-of-04024.json.gz",
        "mc4_ru_00060.json.gz", "MC4 (ru) part 60", 300_000_000),
        # MC4 (be) – 1 шард
        ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-be.tfrecord-00000-of-00016.json.gz",
        "mc4_be_00000.json.gz", "MC4 (be) part 0", 100_000_000),
        # StarCoder (общий)
        ("bigcode/starcoderdata", "data", "train", "content", 1_000_000, "starcoder.jsonl"),
        # CodeParrot (Python)
        ("codeparrot/github-code", "python", "train", "code", 500_000, "codeparrot.jsonl"),
        # Claude Opus 4.7
        ("TeichAI/lordx64-claude-opus-4.7-max-cleaned", "default", "train", "text", 500_000, "claude_opus47.jsonl"),
    ]

    for name, config, split, field, max_rows, local_name in downloads:
        download_dataset(name, config, split, field, max_rows, local_name)

    # ---------- DPO DATASETS ----------
    print("\n" + "=" * 60)
    print("📥 ЗАГРУЗКА DPO DATASETS")
    print("=" * 60)

    dpo_downloads = [
        # UltraFeedback - best quality preference dataset (20k for small model)
        ("argilla/ultrafeedback-binarized-preferences-cleaned", "default", "train", 20_000, "dpo_ultrafeedback.jsonl"),
        # ORPO-DPO Mix - high quality combined dataset (10k for small model)
        ("mlabonne/orpo-dpo-mix-40k-flat", "default", "train", 10_000, "dpo_orpo_mix.jsonl"),
    ]

    for name, config, split, max_rows, local_name in dpo_downloads:
        download_dpo_dataset(name, config, split, max_rows, local_name)

    # Copy DPO files to data/dpo/
    dpo_dest = "data/dpo"
    os.makedirs(dpo_dest, exist_ok=True)
    for fname in ["dpo_ultrafeedback.jsonl", "dpo_orpo_mix.jsonl"]:
        src = os.path.join(TMP, fname)
        if os.path.exists(src):
            import shutil
            dst = os.path.join(dpo_dest, fname)
            shutil.copy(src, dst)
            print(f"  📦 Скопировано: {fname} -> {dpo_dest}")

    print("\n✅ Загрузка DPO завершена!")
    print("\n✅ Вся загрузка завершена. Запустите сборку: python scripts/build_and_tokenize.py")

if __name__ == "__main__":
    main()

## scripts/dpo_train.py

#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import safetensors.torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent


def load_cfg():
    import yaml
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        raw = yaml.safe_load(f) or {}
    merged = {}
    merged.update(raw.get("model", {}))
    merged.update(raw.get("training", {}))
    return merged


def find_sft_checkpoint():
    for f in ["checkpoints/sft/sft_final.safetensors", "checkpoints/sft/sft_best.safetensors"]:
        p = PROJECT_ROOT / f
        if p.exists():
            return str(p)
    return None


class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        cid = tokenizer.chat_ids
        user_tag = cid.get("<|user|>", 0)
        assistant_tag = cid.get("<|assistant|>", 0)

        with open(data_path) as f:
            for line in f:
                r = json.loads(line)
                msgs = r.get("messages", [])
                if len(msgs) < 2:
                    continue

                assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
                if not assistant_msgs:
                    continue

                prompt_ids = []
                for m in msgs:
                    if m.get("role") != "assistant":
                        tag_id = cid.get(f"<|{m['role']}|>", 0)
                        if tag_id:
                            prompt_ids.append(tag_id)
                        content = m.get("content", "")
                        if content:
                            prompt_ids.extend(tokenizer.encode(content))

                chosen = assistant_msgs[0].get("content", "")
                chosen_ids = [assistant_tag] + tokenizer.encode(chosen) if assistant_tag else tokenizer.encode(chosen)

                rejected = "I don't know. Let me think... Actually, I'm not sure about this. Maybe someone else can help? Sorry, I cannot answer this question properly."
                rejected_ids = [assistant_tag] + tokenizer.encode(rejected) if assistant_tag else tokenizer.encode(rejected)

                full_chosen = prompt_ids + chosen_ids
                full_rejected = prompt_ids + rejected_ids

                if len(full_chosen) < 10 or len(full_rejected) < 10:
                    continue
                self.examples.append((full_chosen[:max_len], full_rejected[:max_len]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        c, r = self.examples[idx]
        return torch.tensor(c, dtype=torch.long), torch.tensor(r, dtype=torch.long)


def collate_dpo(batch):
    max_c = max(b[0].size(0) for b in batch)
    max_r = max(b[1].size(0) for b in batch)
    max_len = max(max_c, max_r)
    pad = 0
    chosen_ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    rejected_ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    chosen_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    rejected_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (c, r) in enumerate(batch):
        chosen_ids[i, :c.size(0)] = c
        rejected_ids[i, :r.size(0)] = r
        chosen_mask[i, :c.size(0)] = True
        rejected_mask[i, :r.size(0)] = True
    return chosen_ids, rejected_ids, chosen_mask, rejected_mask


@torch.no_grad()
def get_logprobs(model, input_ids, mask):
    logits, _, _, _ = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_mask = mask[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * shift_mask).sum(dim=-1)


def train_dpo(sft_path: str, data_path: str, output_dir: str,
              lr: float = 5e-6, epochs: int = 2, batch_size: int = 2,
              grad_accum: int = 8, beta: float = 0.1, log_every: int = 10):
    from bulba1.config import ModelConfig
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer

    raw_cfg = load_cfg()
    cfg = ModelConfig(**raw_cfg)
    device = torch.device("cuda")

    tokenizer = FastTokenizer("data/tokenizer_fast.json")
    tokenizer.load()
    tokenizer.add_chat_tokens()

    policy = MiniChat(cfg).to(device).to(torch.bfloat16)
    reference = MiniChat(cfg).to(device).to(torch.bfloat16)

    state = safetensors.torch.load_file(sft_path)
    policy.load_state_dict(state, strict=False)
    reference.load_state_dict(state, strict=False)
    policy.resize_token_embeddings(tokenizer.vocab_size)
    reference.resize_token_embeddings(tokenizer.vocab_size)
    for p in reference.parameters():
        p.requires_grad = False

    dataset = DPODataset(data_path, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_dpo, drop_last=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    total_steps = (len(dataset) // batch_size // grad_accum) * epochs

    os.makedirs(output_dir, exist_ok=True)
    step = 0
    print(f"DPO: {len(dataset)} pairs, {total_steps} steps, beta={beta}")

    policy.train()
    for epoch in range(epochs):
        for i, (chosen_ids, rejected_ids, c_mask, r_mask) in enumerate(loader):
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            c_mask = c_mask.to(device)
            r_mask = r_mask.to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                policy_chosen_lp = get_logprobs(policy, chosen_ids, c_mask)
                policy_rejected_lp = get_logprobs(policy, rejected_ids, r_mask)
                ref_chosen_lp = get_logprobs(reference, chosen_ids, c_mask)
                ref_rejected_lp = get_logprobs(reference, rejected_ids, r_mask)

                policy_ratio = policy_chosen_lp - policy_rejected_lp
                ref_ratio = ref_chosen_lp - ref_rejected_lp
                loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio)).mean()
                loss = loss / grad_accum

            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % log_every == 0:
                    print(f"  step {step:4d}/{total_steps} | loss={loss.item() * grad_accum:.4f}")

    final_path = os.path.join(output_dir, "dpo_final.safetensors")
    safetensors.torch.save_file(policy.state_dict(), final_path)
    print(f"DPO done. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", help="SFT checkpoint path (default: auto-find)")
    parser.add_argument("--data", default="data/sft/sft_claude_opus47.jsonl")
    parser.add_argument("--output", default="checkpoints/dpo")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    sft_ckpt = args.sft_checkpoint or find_sft_checkpoint()
    if not sft_ckpt:
        print("No SFT checkpoint found. Run SFT first.")
        return
    print(f"SFT checkpoint: {sft_ckpt}")
    train_dpo(sft_ckpt, args.data, args.output, args.lr, args.epochs, args.batch_size, args.grad_accum, args.beta)


if __name__ == "__main__":
    main()


## scripts/pretokenize.py

#!/usr/bin/env python3
"""
САМАЯ СТАБИЛЬНАЯ ПРЕДТОКЕНИЗАЦИЯ + YAML + JSONL ЛОГИ
====================================================================
Работает в 1 процесс, но использует 100% ядер процессора за счет Rust.
Идеально сохраняет переносы строк, не падает по памяти.
Выводит логи для Telegram-бота в формате .jsonl.
"""

import argparse
import array
import glob
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import yaml  # Заменили json на yaml для манифеста
from tqdm import tqdm

# Оставляем параллелизм Rust включенным на максимум!
os.environ["TOKENIZERS_PARALLELISM"] = "true"

try:
    from bulba1.tokenizer import FastTokenizer
except ImportError:
    print("WARNING: bulba1.tokenizer not found.")
    FastTokenizer = None

# Читаем стабильными кусками по 50 Мегабайт
CHUNK_SIZE_BYTES = 50 * 1024 * 1024

# ==============================================================================
# Helpers
# ==============================================================================

def guess_domain(filename: str) -> str:
    name = os.path.basename(filename).lower()
    while name.endswith('.txt') or name.endswith('.json'):
        name = name.rsplit('.', 1)[0]
    if 'math' in name or 'metamathqa' in name:
        return 'math'
    base = name.rstrip('0123456789-_')
    prefix = base.split('-')[0].split('_')[0]
    return prefix if prefix else 'unknown'

def sample_indices(n_total: int, n_take: int, seed: int):
    rng = np.random.default_rng(seed)
    if n_take >= n_total:
        return rng.permutation(n_total)
    return rng.choice(n_total, size=n_take, replace=False)

def log_to_jsonl(log_path: str, record: dict):
    """Записывает 1 строчку JSONL для Telegram бота"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ==============================================================================
# Phase 1: Parallel Tokenization (Stable CPU version)
# ==============================================================================

def tokenize_phase(
    data_dir: str,
    tmp_dir: str,
    tokenizer,
    seq_len: int,
    stride: int,
    manifest_path: str,
    log_file: str,
    domain_weights: dict[str, float] | None = None,
):
    txt_files = sorted(glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True))
    if not txt_files:
        print(f"❌ Нет txt‑файлов в {data_dir}")
        sys.exit(1)

    # Читаем YAML манифест
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        existing_files = {entry['path'] for entry in manifest.get('files',[])}
        if not domain_weights and 'domain_weights' in manifest:
            domain_weights = manifest['domain_weights']
    else:
        manifest = {
            'version': 1, 'seq_len': seq_len, 'stride': stride,
            'domain_weights': domain_weights if domain_weights else {}, 'files':[],
        }
        existing_files = set()

    remaining_files =[f for f in txt_files if os.path.relpath(f, start=data_dir) not in existing_files]
    if not remaining_files:
        print("✅ Все файлы уже токенизированы. Пропускаем.")
        return

    print(f"📂 Найдено {len(remaining_files)} новых файлов. Старт стабильной токенизации…")
    os.makedirs(tmp_dir, exist_ok=True)

    new_files_stats =[]

    # Глобальные счетчики для логов бота
    total_bytes_all = sum(os.path.getsize(f) for f in remaining_files)
    processed_bytes_all = 0
    total_tokens_all = sum(e.get('tokens', 0) for e in manifest.get('files',[]))

    # Сбрасываем (очищаем) лог файл перед новым запуском
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    open(log_file, "w").close()

    for txt_path in remaining_files:
        rel_path = os.path.relpath(txt_path, start=data_dir)
        domain = guess_domain(os.path.basename(txt_path))

        domain_bin_path = os.path.join(tmp_dir, f".{domain}.tok.bin")
        file_size_bytes = os.path.getsize(txt_path)

        print(f"\n🔤 Файл: {os.path.basename(txt_path)} ({file_size_bytes / 1024**2:.1f} МБ)")

        file_tokens = 0
        processed_file_bytes = 0

        last_log_time = time.time()
        bytes_since_log = 0
        tokens_since_log = 0

        with open(txt_path, "rb") as fin, \
             open(domain_bin_path, "ab") as fout, \
             tqdm(total=file_size_bytes, unit='B', unit_scale=True, unit_divisor=1024) as pbar:

            while True:
                # Читаем строками, чтобы не рвать слова (ровно ~50 МБ за раз)
                raw_lines = fin.readlines(CHUNK_SIZE_BYTES)
                if not raw_lines:
                    break

                # Декодируем
                str_lines =[line.decode("utf-8", errors="ignore") for line in raw_lines]

                # Токенизируем (Rust съедает это параллельно на всех ядрах)
                encoded = tokenizer.encode_batch(str_lines)

                # Сплющиваем массив
                flattened_ids = itertools.chain.from_iterable(encoded)
                bin_array = array.array('i', flattened_ids)

                # Записываем на диск монолитом
                bin_array.tofile(fout)

                # Обновляем метрики
                chunk_bytes = sum(len(line) for line in raw_lines)
                chunk_tokens = len(bin_array)

                file_tokens += chunk_tokens
                total_tokens_all += chunk_tokens
                processed_file_bytes += chunk_bytes
                processed_bytes_all += chunk_bytes

                bytes_since_log += chunk_bytes
                tokens_since_log += chunk_tokens

                pbar.update(chunk_bytes)
                pbar.set_postfix_str(f"Токены: {file_tokens / 1e6:.2f}M")

                # Логгирование в .jsonl каждую секунду (для Telegram бота)
                now = time.time()
                if now - last_log_time >= 1.0:
                    dt = now - last_log_time
                    mb_per_sec = (bytes_since_log / 1024 / 1024) / dt
                    tok_per_sec = tokens_since_log / dt

                    file_pct = (processed_file_bytes / file_size_bytes) * 100 if file_size_bytes else 100
                    total_pct = (processed_bytes_all / total_bytes_all) * 100 if total_bytes_all else 100

                    log_to_jsonl(log_file, {
                        "task": "pretokenize",
                        "current_file": os.path.basename(txt_path),
                        "file_progress_pct": round(file_pct, 1),
                        "total_progress_pct": round(total_pct, 1),
                        "mb_per_sec": round(mb_per_sec, 2),
                        "tok_per_sec": int(tok_per_sec),
                        "file_tokens_m": round(file_tokens / 1e6, 2),
                        "total_tokens_m": round(total_tokens_all / 1e6, 2),
                        "timestamp": int(now)
                    })

                    bytes_since_log = 0
                    tokens_since_log = 0
                    last_log_time = now

        new_files_stats.append({
            'path': rel_path,
            'domain': domain,
            'tokens': file_tokens,
            'enabled': True
        })

    # Сохраняем YAML манифест
    manifest['files'].extend(new_files_stats)
    if not manifest['domain_weights']:
        domains_in_data = {entry['domain'] for entry in manifest['files'] if entry['enabled']}
        if domains_in_data:
            manifest['domain_weights'] = {d: 1.0 / len(domains_in_data) for d in domains_in_data}
        else:
            manifest['domain_weights'] = {'unknown': 1.0}

    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n✅ Токенизация завершена! Данные сохранены в манифест: {manifest_path}")


# ==============================================================================
# Phase 2: Balance & Shard
# ==============================================================================

def balance_phase(
    manifest_path: str,
    tmp_dir: str,
    output_dir: str,
    num_shards: int,
    seed: int,
    keep_tmp: bool,
) -> list[str]:
    with open(manifest_path, encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    seq_len = manifest['seq_len']
    stride = manifest['stride']
    files = manifest['files']

    domain_tokens = defaultdict(int)
    for entry in files:
        if entry['enabled']:
            domain_tokens[entry['domain']] += entry['tokens']

    domain_samples = {}
    for d, tokens in domain_tokens.items():
        if tokens >= seq_len + 1:
            domain_samples[d] = (tokens - (seq_len + 1)) // stride + 1
        else:
            domain_samples[d] = 0

    active_domains = [d for d in domain_samples if domain_samples[d] > 0]
    if not active_domains:
        raise RuntimeError("No active domains found. Check if tokenization phase produced any data.")

    # Берём 100% данных
    take = {}
    total_picks = 0
    for d in active_domains:
        take[d] = domain_samples[d]
        total_picks += take[d]

    print("\n📊 Состав датасета (используется 100% доступных данных):")
    for d in active_domains:
        share = (take[d] / total_picks) * 100
        print(f"  {d}: {take[d]:,} семплов ({share:.1f}%)")
    print(f"  ИТОГО: {total_picks:,} семплов")

    # Случайные перестановки индексов внутри каждого домена
    domain_indices = {}
    for d in active_domains:
        domain_indices[d] = sample_indices(domain_samples[d], take[d], seed + hash(d) % 100000)

    sample_size = seq_len + 1
    os.makedirs(output_dir, exist_ok=True)

    shard_paths = []
    BUF_SIZE = 50000

    print(f"\n📦 Пишем {total_picks:,} семплов в {num_shards} шард(ов)...")

    domain_mmaps = {}
    for d in active_domains:
        path = os.path.join(tmp_dir, f".{d}.tok.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing domain binary: {path}")
        domain_mmaps[d] = np.memmap(path, dtype=np.int32, mode='r')

    # Генерируем массив позиций для глобального перемешивания
    # Используем np.int64 для sample_idx, чтобы исключить переполнение
    positions = np.empty(total_picks, dtype=[('domain_idx', np.int8), ('sample_idx', np.int64)])
    pos = 0
    for di, d in enumerate(active_domains):
        cnt = len(domain_indices[d])
        positions['domain_idx'][pos:pos+cnt] = di
        positions['sample_idx'][pos:pos+cnt] = domain_indices[d]
        pos += cnt

    # Глобальный шаффл всех доменов между собой
    rng = np.random.default_rng(seed)
    rng.shuffle(positions)

    from contextlib import ExitStack

    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(open(os.path.join(output_dir, f"train_balanced_{i:03d}.bin"), 'wb'))
            for i in range(num_shards)
        ]
        shard_buf_idx = [0] * num_shards
        shard_bufs = [np.empty((BUF_SIZE, sample_size), dtype=np.int32) for _ in range(num_shards)]

        for i in tqdm(range(total_picks), desc="Writing shards"):
            di = positions['domain_idx'][i]
            sample_idx = positions['sample_idx'][i]
            shard_idx = i % num_shards

            mm = domain_mmaps[active_domains[di]]
            # Главное исправление – явный Python int
            offset = int(sample_idx) * stride

            shard_bufs[shard_idx][shard_buf_idx[shard_idx]] = mm[offset:offset + sample_size]
            shard_buf_idx[shard_idx] += 1

            if shard_buf_idx[shard_idx] == BUF_SIZE:
                shard_bufs[shard_idx].tofile(shard_files[shard_idx])
                shard_buf_idx[shard_idx] = 0

        # Сбрасываем остатки буферов
        for shard_idx in range(num_shards):
            if shard_buf_idx[shard_idx] > 0:
                shard_bufs[shard_idx][:shard_buf_idx[shard_idx]].tofile(shard_files[shard_idx])
            shard_paths.append(os.path.join(output_dir, f"train_balanced_{shard_idx:03d}.bin"))

    for mm in domain_mmaps.values():
        mm._mmap.close()

    if not keep_tmp:
        for d in active_domains:
            path = os.path.join(tmp_dir, f".{d}.tok.bin")
            if os.path.exists(path):
                os.remove(path)

    # Обновляем манифест финальной статистикой
    manifest['balanced_output'] = {
        'shards': [os.path.relpath(p, start=output_dir) for p in shard_paths],
        'total_samples': total_picks,
        'seed': seed,
        'proportions': {d: round((take[d]/total_picks)*100, 2) for d in active_domains}
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return shard_paths

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bulba1 preprocessing")
    parser.add_argument('--data-dir', type=str, default='data/train')
    parser.add_argument('--tokenizer-path', type=str, default='data/tokenizer_fast.json')
    parser.add_argument('--output-dir', type=str, default='data/tokenized')
    # Изменили default на .yaml
    parser.add_argument('--manifest', type=str, default='data_manifest.yaml')
    # Путь куда будут писаться логи для бота
    parser.add_argument('--log-file', type=str, default='logs/pretokenize.jsonl')
    parser.add_argument('--tmp-dir', type=str, default=None)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', choices=['tokenize', 'balance', 'all'], default='all')
    parser.add_argument('--keep-tmp', action='store_true')
    parser.add_argument('--domain-weights', type=str, default=None)

    # Игнорируется, так как используем Rust Multi-threading
    parser.add_argument('--num-tokenizer-workers', type=int, default=0, help="Ignored")

    args = parser.parse_args()

    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, '.tmp_domains')
    domain_weights = json.loads(args.domain_weights) if args.domain_weights else None

    if FastTokenizer is None:
        sys.exit(1)

    tokenizer = FastTokenizer(args.tokenizer_path)
    tokenizer.load()

    if args.phase in ('tokenize', 'all'):
        tokenize_phase(
            data_dir=args.data_dir,
            tmp_dir=tmp_dir,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            manifest_path=args.manifest,
            log_file=args.log_file,
            domain_weights=domain_weights,
        )

    if args.phase in ('balance', 'all'):
        if not os.path.exists(args.manifest):
            print("❌ ОШИБКА: Манифест не найден. Сначала запустите фазу tokenize.")
            sys.exit(1)
        balance_phase(
            manifest_path=args.manifest,
            tmp_dir=tmp_dir,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            seed=args.seed,
            keep_tmp=args.keep_tmp,
        )

if __name__ == "__main__":
    main()




## scripts/sft_train.py

#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import safetensors.torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent


def load_cfg():
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        raw = yaml.safe_load(f) or {}
    merged = {}
    merged.update(raw.get("model", {}))
    merged.update(raw.get("training", {}))
    return merged


def find_best_checkpoint():
    cfg = load_cfg()
    ckpt_dir = PROJECT_ROOT / cfg.get("checkpoint_dir", "checkpoints/run_bulba1_67m")
    best = ckpt_dir / "best.safetensors"
    if best.exists():
        return str(best)
    files = sorted(
        ckpt_dir.glob("checkpoint_step_*.safetensors"),
        key=lambda f: int(re.search(r"step_(\d+)", f.name).group(1)),
        reverse=True,
    )
    return str(files[0]) if files else None


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(data_path) as f:
            for line in f:
                r = json.loads(line)
                msgs = r.get("messages", [])
                if len(msgs) >= 2:
                    ids, weights = tokenizer.encode_chat(msgs)
                    if len(ids) >= 10:
                        self.examples.append((ids, weights))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, weights = self.examples[idx]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            weights = weights[: self.max_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        target_ids = torch.tensor(ids, dtype=torch.long)
        loss_weights = torch.tensor(weights, dtype=torch.float32)
        return input_ids, target_ids, loss_weights


def collate(batch):
    max_len = max(b[0].size(0) for b in batch)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    weights = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, (inp, tgt, w) in enumerate(batch):
        n = inp.size(0)
        input_ids[i, :n] = inp
        target_ids[i, :n] = tgt
        weights[i, :n] = w
    return input_ids, target_ids, weights


def train_sft(checkpoint_path: str, data_path: str, output_dir: str,
              lr: float = 2e-5, epochs: int = 3, batch_size: int = 4,
              grad_accum: int = 4, log_every: int = 10):
    from bulba1.config import ModelConfig
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer

    raw_cfg = load_cfg()
    cfg = ModelConfig(**raw_cfg)
    device = torch.device("cuda")

    tokenizer = FastTokenizer("data/tokenizer_fast.json")
    tokenizer.load()
    tokenizer.add_chat_tokens()

    model = MiniChat(cfg).to(device).to(torch.bfloat16)
    state = safetensors.torch.load_file(checkpoint_path)
    model.load_state_dict(state, strict=False)
    model.resize_token_embeddings(tokenizer.vocab_size)

    dataset = SFTDataset(data_path, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    total_steps = (len(dataset) // batch_size // grad_accum) * epochs

    os.makedirs(output_dir, exist_ok=True)
    step = 0
    best_loss = float("inf")

    print(f"SFT: {len(dataset)} examples, {total_steps} steps, {epochs} epochs")
    model.train()
    for epoch in range(epochs):
        accum_loss = 0.0
        for i, (inp, tgt, w) in enumerate(loader):
            inp, tgt, w = inp.to(device), tgt.to(device), w.to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                logits, _, _, _ = model(inp)
                shift_logits = logits[:, :-1, :].reshape(-1, cfg.vocab_size)
                shift_tgt = tgt[:, 1:].reshape(-1)
                shift_w = w[:, 1:].reshape(-1)
                loss = F.cross_entropy(shift_logits, shift_tgt, reduction="none")
                loss = (loss * shift_w).sum() / shift_w.sum().clamp_min(1)
                loss = loss / grad_accum

            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                accum_loss += loss.item() * grad_accum

                if step % log_every == 0:
                    avg = accum_loss / log_every
                    print(f"  step {step:4d}/{total_steps} | loss={avg:.4f}")
                    if avg < best_loss:
                        best_loss = avg
                        save_path = os.path.join(output_dir, "sft_best.safetensors")
                        safetensors.torch.save_file(model.state_dict(), save_path)
                    accum_loss = 0.0

    final_path = os.path.join(output_dir, "sft_final.safetensors")
    safetensors.torch.save_file(model.state_dict(), final_path)
    print(f"Done. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Checkpoint path (default: best)")
    parser.add_argument("--data", default="data/sft/sft_claude_opus47.jsonl")
    parser.add_argument("--output", default="checkpoints/sft")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    if not ckpt:
        print("No checkpoint found.")
        return
    print(f"Checkpoint: {ckpt}")
    train_sft(ckpt, args.data, args.output, args.lr, args.epochs, args.batch_size, args.grad_accum)


if __name__ == "__main__":
    main()


## tests/test_sanity.py

import os
import sys
import tempfile
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.model.mamba import MambaBlock
from bulba1.tokenizer import HFTokenizer, TextDataset, create_dataloader
from bulba1.training.engine import TrainingEngine
from bulba1.training.checkpoint import CheckpointManager
from bulba1.training.chunked_ce import chunked_cross_entropy
from bulba1.training.eval import compute_perplexity


def _make_dummy_data(
    dir_path: str, text: str = "hello world this is test data ", repeats: int = 500
):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "dummy.txt"), "w", encoding="utf-8") as f:
        f.write(text * repeats)


def test_dataset_target_shift():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dummy_data(tmpdir)
        tok = HFTokenizer(vocab_size=100)
        tok.train([os.path.join(tmpdir, "dummy.txt")])
        ds = TextDataset(tok, tmpdir, seq_len=8, return_target=True)
        for input_ids, target_ids in ds:
            assert input_ids.shape == (8,)
            assert target_ids.shape == (8,)
            assert not torch.equal(input_ids, target_ids)
            assert torch.equal(input_ids[1:], target_ids[:-1])
            break
    print("PASS: dataset_target_shift")


def test_chunked_ce_correctness():
    vocab = 1000
    logits = torch.randn(2, 32, vocab, device="cpu")
    targets = torch.randint(0, vocab, (2, 32), device="cpu")
    loss_ref = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
    loss_chunk = chunked_cross_entropy(logits, targets.view(-1), chunk_size=256)
    diff = abs(loss_ref.item() - loss_chunk.item())
    assert diff < 1e-5, (
        f"Chunked CE mismatch: ref={loss_ref.item():.6f} chunk={loss_chunk.item():.6f} diff={diff:.6f}"
    )
    print("PASS: chunked_ce_correctness")


def test_mamba_bf16():
    cfg = ModelConfig(d_model=64, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2)
    m = MambaBlock(cfg).cuda().bfloat16()
    x = torch.randn(1, 8, 64, device="cuda").bfloat16()
    out = m(x)
    assert out.shape == x.shape


def test_eval_device_string():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=500,
        num_experts=4,
        expert_hidden=64,
        seq_len=16,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dummy_data(tmpdir)
        tok = HFTokenizer(vocab_size=500)
        tok.train([os.path.join(tmpdir, "dummy.txt")])
        loader = create_dataloader(
            tok, tmpdir, batch_size=2, seq_len=16, num_workers=0, shuffle=False, return_target=False
        )
        model = MiniChat(cfg).cuda().bfloat16()
        try:
            ppl = compute_perplexity(model, loader, device="cuda", max_batches=2)
            assert isinstance(ppl, float)
            print("PASS: eval_device_string")
        except Exception as e:
            raise AssertionError(f"eval crashed: {e}")


def test_checkpoint_optimizer_ema():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_top_k=2)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()
        saved = mgr.save(model, optimizer, step=1, loss=1.0, ema=None)
        assert saved
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_step_1.safetensors"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_step_1_optimizer.pt"))

        new_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr.load(model, os.path.join(tmpdir, "checkpoint_step_1.safetensors"), optimizer=new_opt)
        for key in optimizer.state_dict()["state"]:
            assert key in new_opt.state_dict()["state"]
    print("PASS: checkpoint_optimizer_ema")


def test_generate_with_kv_cache():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=16,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg).eval()
    x = torch.randint(0, 100, (1, 5))
    out1 = model.generate(x, max_new_tokens=5)
    assert out1.shape == (1, 10)
    out2 = model.generate(x, max_new_tokens=10)
    assert out2.shape == (1, 15)
    print("PASS: generate_with_kv_cache")


def test_training_loss_decreases():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=200,
        num_experts=4,
        expert_hidden=64,
        seq_len=16,
        batch_size=2,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        learning_rate=1e-2,
        use_gradient_checkpointing=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dummy_data(tmpdir, repeats=1000)
        tok = HFTokenizer(vocab_size=200)
        tok.train([os.path.join(tmpdir, "dummy.txt")])
        loader = create_dataloader(tok, tmpdir, cfg.batch_size, cfg.seq_len, num_workers=0)
        model = MiniChat(cfg)
        engine = TrainingEngine(model, cfg, tok, device="cpu")
        # disable EMA for deterministic CPU test
        engine.ema = None

        def inf():
            while True:
                for batch in loader:
                    yield tuple(b for b in batch)

        losses = []
        for step in range(20):
            for accum_step in range(engine.grad_accum_steps):
                batch = next(inf())
                is_last = accum_step == engine.grad_accum_steps - 1
                metrics = engine.train_step(batch, is_accum_last=is_last)
            losses.append(metrics["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
        )
        print(f"PASS: training_loss_decreases ({losses[0]:.4f} -> {losses[-1]:.4f})")


def test_checkpoint_find_latest():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_top_k=5)
        for step in [10, 20, 30]:
            mgr.save(model, optimizer, step=step, loss=1.0 / step, ema=None)
        latest = mgr.find_latest()
        assert "step_30" in latest
    print("PASS: checkpoint_find_latest")


def test_oom_recovery_batch_reduction():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=200,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        batch_size=4,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    engine = TrainingEngine(model, cfg, None, device="cpu")
    engine.cfg.batch_size = 4
    engine.grad_accum_steps = 1
    engine._reduce_batch_size()
    assert engine.cfg.batch_size == 2
    assert engine.grad_accum_steps == 2
    print("PASS: oom_recovery_batch_reduction")


def test_resume_from_checkpoint():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_top_k=2)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()
        mgr.save(model, optimizer, step=42, loss=1.0, ema=None)

        new_model = MiniChat(cfg)
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        loaded_step = mgr.load(new_model, "latest", optimizer=new_opt)
        assert loaded_step == 42
    print("PASS: resume_from_checkpoint")


def test_use_rex_without_moe():
    """Regression: use_rex=True + use_moe=False should not crash with AttributeError."""
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_moe=False,
        use_rex=True,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits, _, _, _ = model(x)
    assert logits.shape == (1, 8, 100)
    print("PASS: use_rex_without_moe")


def run_all():
    test_dataset_target_shift()
    test_chunked_ce_correctness()
    test_mamba_bf16()
    test_eval_device_string()
    test_checkpoint_optimizer_ema()
    test_checkpoint_find_latest()
    test_oom_recovery_batch_reduction()
    test_resume_from_checkpoint()
    test_generate_with_kv_cache()
    test_training_loss_decreases()
    test_use_rex_without_moe()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    run_all()


## tools/deep_profile.py

#!/usr/bin/env python3
"""
Bulba1 Deep Profiler - Comprehensive training performance analysis.
Run with: uv run python tools/deep_profile.py
"""
import os
import sys
import time
import json
import math
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.training.optimizer import CombinedOptimizer


class DeepProfiler:
    def __init__(self, cfg_path="configs/default.yaml", batch_size=4, seq_len=256, warmup_steps=2, profile_steps=5):
        self.cfg_path = cfg_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.results = {"config": {}, "components": {}, "recommendations": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.cfg = None

    def load_config(self):
        with open(self.cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        all_params = {}
        all_params.update(yaml_cfg.get('model', {}))
        all_params.update(yaml_cfg.get('training', {}))
        self.cfg = ModelConfig(**all_params)
        self.results["config"] = {
            "d_model": self.cfg.d_model,
            "n_layers": self.cfg.n_layers,
            "n_heads": self.cfg.n_heads,
            "vocab_size": self.cfg.vocab_size,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "use_gradient_checkpointing": self.cfg.use_gradient_checkpointing,
            "use_mhc": self.cfg.use_mhc,
            "use_moe": self.cfg.use_moe,
            "use_bitnet_a48": self.cfg.use_bitnet_a48,
            "use_bitlinear": self.cfg.use_bitlinear,
            "muon_ns_steps": self.cfg.muon_ns_steps,
        }

    def init_model(self):
        print("  Initializing model...")
        self.model = MiniChat(self.cfg).to(self.device)
        self.model.train()
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.results["model_info"] = {
            "total_params_m": round(param_count / 1e6, 2),
            "trainable_params_m": round(trainable_count / 1e6, 2),
        }
        print(f"    Params: {param_count/1e6:.1f}M")

    def init_optimizer(self):
        print("  Initializing optimizer...")
        self.optimizer = CombinedOptimizer(self.model, self.cfg)
        muon_count = len(self.optimizer.muon.param_groups[0]["params"]) if self.optimizer.muon else 0
        adamw_count = len(self.optimizer.adamw.param_groups[0]["params"]) if self.optimizer.adamw else 0
        self.results["optimizer_info"] = {"muon_params": muon_count, "adamw_params": adamw_count}
        print(f"    Muon: {muon_count}, AdamW: {adamw_count}")

    def profile_full_step(self):
        print("\n🔄 FULL TRAINING STEP PROFILE")
        print("=" * 60)

        times = {"forward": [], "backward": [], "optimizer": [], "total": []}
        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = x.clone()

        for step in range(self.warmup_steps + self.profile_steps):
            try:
                self.model.zero_grad()
                self.optimizer.zero_grad()
                torch.cuda.synchronize()

                t0 = time.perf_counter()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _, _, aux_loss = self.model(x, 1)

                text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
                loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))
                loss = loss + aux_loss * 0.001

                torch.cuda.synchronize()
                t_fwd = time.perf_counter()

                loss.backward()

                torch.cuda.synchronize()
                t_bwd = time.perf_counter()

                self.optimizer.step()

                torch.cuda.synchronize()
                t_opt = time.perf_counter()

                if step >= self.warmup_steps:
                    times["forward"].append(t_fwd - t0)
                    times["backward"].append(t_bwd - t_fwd)
                    times["optimizer"].append(t_opt - t_bwd)
                    times["total"].append(t_opt - t0)

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error at step {step}: {e}")
                torch.cuda.empty_cache()
                break

        if not times["total"]:
            print("  Unable to profile - errors encountered")
            return 0

        avg_total = sum(times["total"]) / len(times["total"]) * 1000
        avg_fwd = sum(times["forward"]) / len(times["forward"]) * 1000
        avg_bwd = sum(times["backward"]) / len(times["backward"]) * 1000
        avg_opt = sum(times["optimizer"]) / len(times["optimizer"]) * 1000

        self.results["step_profile"] = {
            "total_ms": round(avg_total, 2),
            "forward_ms": round(avg_fwd, 2),
            "backward_ms": round(avg_bwd, 2),
            "optimizer_ms": round(avg_opt, 2),
            "pct_forward": round(avg_fwd / avg_total * 100, 1),
            "pct_backward": round(avg_bwd / avg_total * 100, 1),
            "pct_optimizer": round(avg_opt / avg_total * 100, 1),
        }

        print(f"  Total:       {avg_total:>8.2f} ms  (100.0%)")
        print(f"  Forward:     {avg_fwd:>8.2f} ms  ({self.results['step_profile']['pct_forward']:>5.1f}%)")
        print(f"  Backward:    {avg_bwd:>8.2f} ms  ({self.results['step_profile']['pct_backward']:>5.1f}%)")
        print(f"  Optimizer:   {avg_opt:>8.2f} ms  ({self.results['step_profile']['pct_optimizer']:>5.1f}%)")

        return avg_total

    def profile_memory(self):
        print("\n📊 MEMORY PROFILE")
        print("=" * 60)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        baseline_mem = torch.cuda.memory_allocated() / 1024**2

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = self.model(x, 1)

            forward_mem = torch.cuda.max_memory_allocated() / 1024**2

            targets = x.clone()
            text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
            loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))
            loss.backward()

            backward_mem = torch.cuda.max_memory_allocated() / 1024**2

            self.optimizer.step()
            self.optimizer.zero_grad()

            after_opt_mem = torch.cuda.memory_allocated() / 1024**2
        except Exception as e:
            print(f"  Memory test error: {e}")
            forward_mem = backward_mem = after_opt_mem = 0

        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

        self.results["memory"] = {
            "baseline_mb": round(baseline_mem, 1),
            "forward_mb": round(forward_mem, 1),
            "backward_mb": round(backward_mem, 1),
            "after_optimizer_mb": round(after_opt_mem, 1),
            "total_vram_gb": round(total_vram, 1),
            "utilization_pct": round(forward_mem / (total_vram * 1024) * 100, 1),
        }

        print(f"  Baseline:    {baseline_mem:>8.1f} MB")
        print(f"  Forward:     {forward_mem:>8.1f} MB")
        print(f"  Backward:   {backward_mem:>8.1f} MB")
        print(f"  After Opt:   {after_opt_mem:>8.1f} MB")
        print(f"  Peak VRAM:  {forward_mem:>8.1f} MB ({self.results['memory']['utilization_pct']}%)")

    def profile_block_breakdown(self):
        print("\n🔧 BLOCK COMPONENT PROFILING")
        print("=" * 60)

        block_times = []

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        h = self.model.embedding(x)
        if self.cfg.num_clr_tokens > 0:
            clr = self.model.clr_tokens.expand(self.batch_size, -1, -1)
            h = torch.cat([clr, h], dim=1)

        for _ in range(self.warmup_steps):
            self.model.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for block in self.model.blocks:
                    h, _, _ = block(h, None)

        for i, block in enumerate(self.model.blocks):
            self.model.zero_grad()
            h_temp = h.clone()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    h_out, _, _ = block(h_temp, None)
                torch.cuda.synchronize()
                block_ms = (time.perf_counter() - t0) * 1000

                block_times.append({
                    "layer": i,
                    "is_attn": block.is_attn_block,
                    "ms": round(block_ms, 2),
                })
                print(f"  Block {i:2d} ({'Attn' if block.is_attn_block else 'Mamba'}): {block_ms:>8.2f} ms")

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Block {i} error: {e}")
                break

        if block_times:
            total_block_time = sum(b["ms"] for b in block_times)
            self.results["blocks"] = {
                "total_ms": round(total_block_time, 2),
                "per_block_avg_ms": round(total_block_time / len(block_times), 2),
                "details": block_times,
            }

    def analyze_activations(self):
        print("\n🔥 ACTIVATION SCALE ANALYSIS")
        print("=" * 60)

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        h = self.model.embedding(x)
        if self.cfg.num_clr_tokens > 0:
            clr = self.model.clr_tokens.expand(self.batch_size, -1, -1)
            h = torch.cat([clr, h], dim=1)

        scales = []

        for i, block in enumerate(self.model.blocks):
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    h, _, _ = block(h, None)

                scale = h.abs().max().item()
                scales.append(scale)

                if i < 3 or i >= self.cfg.n_layers - 2 or i % 5 == 0:
                    status = ""
                    if scale > 10000:
                        status = " ⚠️ EXPLOSION"
                    print(f"  Block {i:2d}: scale={scale:>12.2f}{status}")
            except Exception as e:
                print(f"  Block {i} error: {e}")
                break

        if scales:
            self.results["activations"] = {
                "initial_scale": round(scales[0], 2),
                "max_scale": round(max(scales), 2),
                "final_scale": round(scales[-1], 2),
                "scale_growth": round(max(scales) / scales[0], 2) if scales[0] > 0 else 0,
                "has_explosion": max(scales) > 10000,
            }
            print(f"\n  Initial: {scales[0]:.2f}")
            print(f"  Max:     {max(scales):.2f}")
            print(f"  Final:   {scales[-1]:.2f}")
            print(f"  Growth:  {self.results['activations']['scale_growth']:.1f}x")

            if self.results["activations"]["has_explosion"]:
                print("  ⚠️  Activation explosion detected!")

    def analyze_gradients(self):
        print("\n📈 GRADIENT ANALYSIS")
        print("=" * 60)

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = x.clone()

        self.model.zero_grad()
        self.optimizer.zero_grad()

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, aux_loss = self.model(x, 1)

            text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
            loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))
            loss = loss + aux_loss * 0.001

            loss.backward()

            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()

            sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:10]

            self.results["gradients"] = {
                "top_grads": [(n, round(g, 4)) for n, g in sorted_grads],
                "has_nan_grads": any(math.isnan(g) for g in grad_norms.values()),
                "total_params_with_grad": len(grad_norms),
            }

            print("  Top 10 gradient norms:")
            for name, grad in sorted_grads:
                short_name = name if len(name) < 50 else "..." + name[-47:]
                print(f"    {short_name:<50} {grad:>10.4f}")

            if self.results["gradients"]["has_nan_grads"]:
                print("  🔴 NaN gradients detected!")

        except Exception as e:
            print(f"  Gradient analysis error: {e}")

    def analyze_loss(self):
        print("\n💰 LOSS ANALYSIS")
        print("=" * 60)

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = x.clone()

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, mtp1, mtp2, aux_loss = self.model(x, 1)

            text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
            main_loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))

            self.results["loss_analysis"] = {
                "main_loss": round(main_loss.item(), 4),
                "aux_loss": round(aux_loss.item(), 4),
                "logits_scale": round(logits.abs().max().item(), 2),
                "has_nan": torch.isnan(logits).any().item(),
            }

            print(f"  Main loss:   {main_loss.item():.4f}")
            print(f"  Aux loss:    {aux_loss.item():.4f}")
            print(f"  Logits max:  {logits.abs().max().item():.2f}")

            if self.results["loss_analysis"]["has_nan"]:
                print("  🔴 NaN in logits!")

        except Exception as e:
            print(f"  Loss analysis error: {e}")

    def compute_throughput(self):
        print("\n⚡ THROUGHPUT ANALYSIS")
        print("=" * 60)

        if "step_profile" not in self.results:
            print("  No step profile data")
            return

        total_ms = self.results["step_profile"]["total_ms"]
        tokens_per_batch = self.batch_size * self.seq_len
        tokens_per_sec = tokens_per_batch / (total_ms / 1000) if total_ms > 0 else 0

        steps_per_day = (24 * 60 * 60) / (total_ms / 1000) if total_ms > 0 else 0
        days_for_100k = 100000 / steps_per_day if steps_per_day > 0 else float('inf')

        self.results["throughput"] = {
            "tokens_per_sec": round(tokens_per_sec, 1),
            "ms_per_step": round(total_ms, 1),
            "steps_per_day": round(steps_per_day, 0),
            "days_for_100k": round(days_for_100k, 1),
        }

        print(f"  Tokens/sec:      {tokens_per_sec:>8.1f}")
        print(f"  Time/step:       {total_ms:>8.1f} ms")
        print(f"  Steps/day:       {steps_per_day:>8.0f}")
        print(f"  100K steps:      {days_for_100k:>8.1f} days")

    def generate_recommendations(self):
        print("\n💡 RECOMMENDATIONS")
        print("=" * 60)

        recs = []

        if self.results.get("memory", {}).get("utilization_pct", 0) > 85:
            recs.append(("CRITICAL", "High VRAM usage. Consider reducing batch size."))

        if self.results.get("activations", {}).get("has_explosion"):
            recs.append(("CRITICAL", "Activation explosion! Check normalization and residual scaling."))

        if self.results.get("gradients", {}).get("has_nan_grads"):
            recs.append(("CRITICAL", "NaN gradients detected. Training is unstable."))

        if self.results.get("loss_analysis", {}).get("has_nan"):
            recs.append(("CRITICAL", "NaN in logits. Check model initialization."))

        step = self.results.get("step_profile", {})
        if step.get("pct_optimizer", 0) > 30:
            recs.append(("HIGH", f"Optimizer is {step['pct_optimizer']}% of step. Consider reducing muon_ns_steps."))

        if step.get("pct_backward", 0) > 50:
            recs.append(("MEDIUM", f"Backward pass is {step['pct_backward']}% of step. Consider gradient checkpointing."))

        tp = self.results.get("throughput", {}).get("tokens_per_sec", 0)
        if tp > 0 and tp < 50:
            recs.append(("MEDIUM", f"Low throughput ({tp:.1f} tok/s). Check for bottlenecks."))

        if self.cfg.use_mhc:
            recs.append(("INFO", f"MHC enabled: n={self.cfg.mhc_n}, iterations={self.cfg.mhc_iterations}"))

        if self.cfg.use_gradient_checkpointing:
            recs.append(("INFO", "Gradient checkpointing is enabled"))

        self.results["recommendations"] = [{"priority": p, "text": t} for p, t in recs]

        for priority, text in recs:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🔵"}.get(priority, "⚪")
            print(f"  {icon} {priority}: {text}")

    def save_results(self):
        output_path = Path("logs/deep_profile.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)

        model_info = self.results.get("model_info", {})
        mem = self.results.get("memory", {})
        tp = self.results.get("throughput", {})

        print(f"""
  Model:       {model_info.get('total_params_m', 'N/A')}M params
  VRAM:        {mem.get('forward_mb', 'N/A')} MB ({mem.get('utilization_pct', 'N/A')}% utilized)
  Throughput:  {tp.get('tokens_per_sec', 'N/A')} tokens/sec
  Time/step:   {tp.get('ms_per_step', 'N/A')} ms
  100K steps:  {tp.get('days_for_100k', 'N/A')} days
""")

    def run(self):
        print("\n" + "=" * 60)
        print("🚀 BULBA1 DEEP PROFILER")
        print("=" * 60)

        self.load_config()
        self.init_model()
        self.init_optimizer()

        print(f"\n  Config: {self.cfg_path}")
        print(f"  Batch: {self.batch_size}, Seq: {self.seq_len}")
        print(f"  Model: {self.results['model_info']['total_params_m']}M params")
        print(f"  Device: {self.device}")

        try:
            self.profile_memory()
        except Exception as e:
            print(f"  Memory profile error: {e}")

        try:
            self.profile_full_step()
        except Exception as e:
            print(f"  Step profile error: {e}")

        try:
            self.profile_block_breakdown()
        except Exception as e:
            print(f"  Block breakdown error: {e}")

        try:
            self.analyze_activations()
        except Exception as e:
            print(f"  Activation analysis error: {e}")

        try:
            self.analyze_gradients()
        except Exception as e:
            print(f"  Gradient analysis error: {e}")

        try:
            self.analyze_loss()
        except Exception as e:
            print(f"  Loss analysis error: {e}")

        try:
            self.compute_throughput()
        except Exception as e:
            print(f"  Throughput error: {e}")

        self.generate_recommendations()
        self.save_results()
        self.print_summary()

        return self.results


def main():
    parser = argparse.ArgumentParser(description="Bulba1 Deep Profiler")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--batch", "-b", type=int, default=4)
    parser.add_argument("--seq", "-s", type=int, default=256)
    parser.add_argument("--warmup", "-w", type=int, default=2)
    parser.add_argument("--profile", "-p", type=int, default=5)
    args = parser.parse_args()

    profiler = DeepProfiler(
        cfg_path=args.config,
        batch_size=args.batch,
        seq_len=args.seq,
        warmup_steps=args.warmup,
        profile_steps=args.profile,
    )

    profiler.run()


if __name__ == "__main__":
    main()

## tools/log_viz.py

#!/usr/bin/env python3
"""
Расширенный визуализатор логов Bulba 1 (JSONL)
Улучшенная информативность: тренд loss, больше метрик в сводке.
"""

import os, sys, glob, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#adb5bd',
})


def find_log_files(logs_dir="logs"):
    patterns = [os.path.join(logs_dir, "*.jsonl"), os.path.join(logs_dir, "**", "*.jsonl")]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)


def select_log_file(files):
    if not files:
        print("No .jsonl files found")
        return None
    print("\nFound log files:")
    print("-" * 60)
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f)
        sz = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024**2:.1f} MB"
        print(f"  [{i}] {os.path.basename(f):<40} ({sz})")
    print("-" * 60)
    while True:
        try:
            choice = input(f"Select file (1-{len(files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            return None


def parse_log(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                ts = rec.get("timestamp", "")
                if " " in ts:
                    time_part = ts.split()[1]
                else:
                    time_part = ts
                data.append({
                    "time": time_part,
                    "step": int(rec["step"]),
                    "total_steps": int(rec["total_steps"]),
                    "loss": float(rec["loss"]),
                    "ema_loss": float(rec.get("ema_loss", 0)),
                    "best_loss": float(rec.get("best_loss", 0)),
                    "lr": float(rec["lr"]),
                    "stage": rec.get("stage", "unknown"),
                    "optimizer": rec.get("optimizer", "Muon+AdamW"),
                    "vram_used": int(rec["vram_used_mb"]),
                    "vram_total": int(rec["vram_total_mb"]),
                    "vram_pct": float(rec.get("vram_pct", 0)),
                    "ram_used": int(rec["ram_used_mb"]),
                    "ram_total": int(rec["ram_total_mb"]),
                    "ram_pct": float(rec.get("ram_pct", 0)),
                    "cpu_pct": int(rec["cpu_pct"]),
                    "tok_per_s": int(rec["tok_per_sec"]),
                    "oom": int(rec["oom_count"]),
                    "batch": int(rec.get("batch_size", 0)),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    if not data:
        print("No valid records found")
        return None
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} steps")
    return df


def generate_plots(df, output_path, title=None):
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        "loss": "#4263eb", "loss_trend": "#f03c3c", "ema": "#e8590c",
        "lr": "#2b8a3e", "tok": "#845ef7", "vram": "#fd7e14",
        "ram": "#fab005", "cpu": "#7950f2", "batch": "#e64980",
        "ma": "#f03c3c", "trend": "#0c8599",
    }

    # 1. Loss + EMA + Best + Trend lines
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["step"], df["loss"], color=colors["loss"], lw=1.2, alpha=0.7, label="Loss")
    ax1.plot(df["step"], df["ema_loss"], color=colors["ema"], lw=1.4, alpha=0.9, label="EMA Loss")
    ax1.axhline(df["best_loss"].min(), color=colors["loss_trend"], ls="--", lw=1.2, label="Best")
    # Trend line (linear fit)
    if len(df) > 2:
        z = np.polyfit(df["step"], df["loss"], 1)
        p = np.poly1d(z)
        ax1.plot(df["step"], p(df["step"]), "-", color=colors["trend"], lw=1.8, alpha=0.8,
                 label=f"Trend ({z[0]:.2e}/step)")
        ax1.text(0.02, 0.98, f"Slope: {z[0]:.2e}",
                 transform=ax1.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.set_title("Loss, EMA & Trend")
    ax1.legend(ncol=2, loc="upper right")

    # 2. Loss with Moving Average
    ax2 = fig.add_subplot(gs[0, 1])
    w = min(10, max(3, len(df)//10))
    df["loss_ma"] = df["loss"].rolling(window=w, center=True, min_periods=1).mean()
    ax2.plot(df["step"], df["loss"], color=colors["loss"], alpha=0.25, lw=0.8, label="Raw")
    ax2.plot(df["step"], df["loss_ma"], color=colors["ma"], lw=2, label=f"MA({w})")
    ax2.set_title("Loss with Moving Average")
    ax2.legend()

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df["step"], df["lr"], color=colors["lr"], lw=1.5)
    ax3.set_title("Learning Rate")
    ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))

    # 4. Tokens/sec
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(df["step"], df["tok_per_s"], color=colors["tok"], lw=0.8, alpha=0.4, label="tok/s")
    ax4.plot(df["step"], df["tok_per_s"].rolling(window=w, center=True, min_periods=1).mean(),
             color=colors["tok"], lw=1.8, label=f"MA({w})")
    ax4.set_ylim(df["tok_per_s"].min()*0.95, df["tok_per_s"].max()*1.02)
    ax4.set_title("Training Speed")
    ax4.legend()

    # 5. VRAM
    ax5 = fig.add_subplot(gs[1, 0])
    vram_used_gb = df["vram_used"] / 1024
    vram_total_gb = df["vram_total"] / 1024
    ax5.plot(df["step"], vram_used_gb, color=colors["vram"], lw=1.5)
    ax5.axhline(vram_total_gb.iloc[0], color="gray", ls="--", alpha=0.5)
    ax5.fill_between(df["step"], vram_used_gb, alpha=0.1, color=colors["vram"])
    ax5.set_ylim(vram_used_gb.min()*0.9, vram_total_gb.iloc[0]*1.05)
    ax5.set_title("VRAM (GB)")

    # 6. RAM
    ax6 = fig.add_subplot(gs[1, 1])
    ram_used_gb = df["ram_used"] / 1024
    ram_total_gb = df["ram_total"] / 1024
    ax6.plot(df["step"], ram_used_gb, color=colors["ram"], lw=1.5)
    ax6.axhline(ram_total_gb.iloc[0], color="gray", ls="--", alpha=0.5)
    ax6.fill_between(df["step"], ram_used_gb, alpha=0.1, color=colors["ram"])
    ax6.set_ylim(ram_used_gb.min()*0.9, ram_total_gb.iloc[0]*1.05)
    ax6.set_title("RAM (GB)")

    # 7. CPU + OOM
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(df["step"], df["cpu_pct"], color=colors["cpu"], lw=1.5, label="CPU")
    ax7.set_ylim(0, 105)
    ax7.set_title("CPU / OOM")
    oom_steps = df[df["oom"] > 0]["step"]
    if len(oom_steps):
        ax7.scatter(oom_steps, [90]*len(oom_steps), color="red", marker="x", s=50, label="OOM")

    # 8. Batch Size
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(df["step"], df["batch"], color=colors["batch"], lw=1.5, drawstyle="steps-post")
    ax8.set_title("Batch Size")
    ax8.set_ylim(bottom=0)

    # 9. Loss Histogram
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.hist(df["loss"], bins=min(30, len(df)//3), color=colors["loss"], alpha=0.6, edgecolor="white")
    ax9.set_title("Loss Distribution")

    # 10. Tok/s Distribution
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(df["tok_per_s"], bins=min(30, len(df)//3), color=colors["tok"], alpha=0.6, edgecolor="white")
    ax10.set_title("Tok/s Distribution")

    # 11. EMA vs Loss scatter (new)
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.scatter(df["loss"], df["ema_loss"], c=df["step"], cmap="viridis", alpha=0.5, s=10)
    ax11.plot([df["loss"].min(), df["loss"].max()], [df["loss"].min(), df["loss"].max()],
              "r--", lw=0.8, alpha=0.5)
    ax11.set_xlabel("Loss")
    ax11.set_ylabel("EMA Loss")
    ax11.set_title("EMA vs Loss")

    # 12. Summary (enriched)
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis("off")
    last = df.iloc[-1]
    progress = 100.0 * last["step"] / last["total_steps"] if last["total_steps"] > 0 else 0
    # Header
    ax12.text(0.5, 0.95, f"PHASE: {last['stage']}", transform=ax12.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="top",
              color="#1e40af", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe", edgecolor="#1e40af"))
    # Progress bar
    ax12.barh(0.80, progress/100.0, height=0.06, color=colors["loss"], alpha=0.7, transform=ax12.transAxes)
    ax12.barh(0.80, 1.0, height=0.06, color="none", edgecolor="#94a3b8", lw=1, transform=ax12.transAxes)
    ax12.text(0.5, 0.80, f"{progress:.1f}%", transform=ax12.transAxes, fontsize=10, ha="center", va="center")

    # Metrics
    loss_change = last["loss"] - df["loss"].iloc[0]
    trend_str = ""
    if len(df) > 2:
        z = np.polyfit(df["step"], df["loss"], 1)
        trend_str = f" | Trend: {z[0]:.2e}/step"
    ax12.text(0.5, 0.64, f"Loss: {last['loss']:.4f} | Best: {df['loss'].min():.4f}{trend_str}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.56, f"EMA: {last['ema_loss']:.4f} | Mean: {df['loss'].mean():.4f}\n"
              f"Change: {loss_change:+.4f} | Tokens: {df['step'].iloc[-1] - df['step'].iloc[0]}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.42, f"Speed: {df['tok_per_s'].mean():.0f} tok/s | Peak: {df['tok_per_s'].max()}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.32, f"VRAM: {last['vram_used']/1024:.1f}GB | RAM: {last['ram_used']/1024:.1f}GB\n"
              f"CPU: {last['cpu_pct']}% | Batch: {last['batch']}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.18, f"OOM: {int(df['oom'].sum())} | LR: {last['lr']:.2e}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top",
              color="#dc2626" if int(df['oom'].sum()) > 0 else "#16a34a")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Bulba 1 Training Visualizer (JSONL)")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output", "-o")
    parser.add_argument("--file", "-f")
    args = parser.parse_args()

    if args.file:
        log_file = args.file
        if not os.path.exists(log_file):
            print(f"File not found: {log_file}")
            return
    else:
        files = find_log_files(args.logs_dir)
        log_file = select_log_file(files)
        if not log_file:
            return

    print(f"\nSelected: {log_file}")
    df = parse_log(log_file)
    if df is None:
        return

    output = args.output or f"{Path(log_file).stem}_dashboard.png"
    title = f"Bulba 1 Training: {Path(log_file).stem.replace('_', ' ').title()}"
    result = generate_plots(df, output, title)
    print(f"\nSaved: {os.path.abspath(result)}")
    print(f"Steps: {len(df):,} | Loss: {df['loss'].iloc[0]:.3f} -> {df['loss'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()


---

*Total: 293.1 KB of code*
