import math
import torch
from torch.optim import Optimizer


class MuonOptimizer(Optimizer):
    """Muon optimizer with momentum and per-parameter RMS scaling.

    Key improvements from "Muon is Scalable for LLM Training" (Moonlight paper):
    1. Momentum: EMA of gradients before Newton-Schulz
    2. Per-parameter update scale: rescale Newton-Schulz output to match grad RMS
    3. Weight decay: standard decoupled weight decay (critical for scale)
    """

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
                    update = self.newton_schulz(g, ns_steps)

                    A, B = g.shape
                    scale = 0.2 * math.sqrt(max(A, B))
                    update.mul_(scale)
                else:
                    update = g

                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss

    def newton_schulz(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        a, b, c = (3.4445, -4.7750, 2.0315)
        # Use bfloat16 for speed on CUDA (Newton-Schulz is compute-bound)
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
    """Routes 2D weight matrices to Muon, everything else to AdamW.

    Per Moonlight paper:
    - Muon: all 2D weight matrices (W_q, W_k, W_v, W_o, W_moe, W_ffn, etc.)
    - AdamW: embeddings, lm_head, biases, norms, non-matrix params
    """

    def __init__(self, model, cfg):
        muon_params = []
        adamw_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Muon: 2D matrices with both dims >= 2 (all weight matrices)
            # AdamW: embeddings, head, biases, norms, 1D/ND params
            is_2d_matrix = p.dim() == 2 and min(p.size(0), p.size(1)) >= 2
            is_embedding_or_head = (
                "embed" in name or "head" in name or "lm_" in name or "bias" in name
            )
            if is_2d_matrix and not is_embedding_or_head:
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
                fused=(
                    cfg.use_f16
                    and hasattr(torch.cuda, "is_available")
                    and torch.cuda.is_available()
                ),
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
