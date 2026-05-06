import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.bit_linear import make_linear


def _parallel_scan_affine(a, B):
    """Parallel prefix scan for affine recurrence S_t = a_t * S_{t-1} + B_t.

    Uses iterative doubling (Hillis-Steele) with associative operator:
        (a1, B1) ⊗ (a2, B2) = (a1*a2, a2*B1 + B2)

    Runs in O(log T) sequential steps, each fully vectorized over GPU.
    """
    B_batch, T, H, D, _ = B.shape
    if T == 1:
        return B

    a_cum = a.clone()
    B_cum = B.clone()

    step = 1
    while step < T:
        a_left = torch.cat(
            [
                torch.ones(B_batch, step, H, 1, 1, device=a.device, dtype=a.dtype),
                a_cum[:, :-step],
            ],
            dim=1,
        )
        B_left = torch.cat(
            [
                torch.zeros(B_batch, step, H, D, D, device=B.device, dtype=B.dtype),
                B_cum[:, :-step],
            ],
            dim=1,
        )

        a_old = a_cum
        B_old = B_cum

        a_cum = a_left * a_old
        B_cum = a_old * B_left + B_old

        step *= 2

    return B_cum


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.gate_dim = getattr(cfg, "kda_gate_dim", 16)
        self.use_parallel_scan = getattr(cfg, "kda_use_parallel_scan", True)

        self.q_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)

        self.gate_proj = make_linear(cfg, cfg.d_model, self.n_heads * self.gate_dim, bias=False)
        self.gate_out = nn.Linear(self.gate_dim, 1, bias=False)

        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

    def _forward_sequential(self, q, k, v, gate):
        B, H, T, D = q.shape
        S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        ys = []
        for t in range(T):
            k_t = k[:, :, t, :].unsqueeze(-1)
            v_t = v[:, :, t, :].unsqueeze(-2)
            g = gate[:, t].unsqueeze(-1).unsqueeze(-1)
            S = g * S + (1 - g) * (k_t @ v_t)
            q_t = q[:, :, t, :].unsqueeze(-1)
            y = (q_t.transpose(-2, -1) @ S).squeeze(-2)
            ys.append(y)
        return torch.stack(ys, dim=2)

    def _forward_parallel(self, q, k, v, gate):
        B, H, T, D = q.shape
        g = gate.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        kv = k.unsqueeze(-1) @ v.unsqueeze(-2)
        a = g
        B_mat = (1 - g) * kv

        a = a.permute(0, 2, 1, 3, 4)
        B_mat = B_mat.permute(0, 2, 1, 3, 4)

        S = _parallel_scan_affine(a, B_mat)
        S = S.permute(0, 2, 1, 3, 4)

        q_exp = q.unsqueeze(-1)
        out = (q_exp.transpose(-2, -1) @ S).squeeze(-2)
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

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        gate_logits = self.gate_proj(x).view(B, T, H, self.gate_dim)
        gate = torch.sigmoid(self.gate_out(gate_logits).squeeze(-1))

        if self.use_parallel_scan and T > 1:
            out = self._forward_parallel(q, k, v, gate)
        else:
            out = self._forward_sequential(q, k, v, gate)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), None, torch.tensor(0.0, device=x.device)
