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
