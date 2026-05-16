import math
import torch
from torch.optim import Optimizer


class MuonOptimizer(Optimizer):
    """Muon optimizer with momentum, per-parameter RMS scaling, and compiled Newton-Schulz."""

    def __init__(
        self, params, lr=3e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5, min_dim=2
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            min_dim=min_dim,
        )
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