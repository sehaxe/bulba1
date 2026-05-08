import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.bit_linear import make_linear, quantize_ste_absmax


def _parallel_scan_affine(a, B):
    B_batch, T, H, D, _ = B.shape
    if T == 1:
        return B
    a_cum = a.clone()
    B_cum = B.clone()
    step = 1
    while step < T:
        a_left = a_cum.roll(step, dims=1)
        B_left = B_cum.roll(step, dims=1)
        a_left[:, :step] = 1.0
        B_left[:, :step] = 0.0
        a_old = a_cum
        B_old = B_cum
        a_cum = a_left * a_old
        B_cum = a_old * B_left + B_old
        step *= 2
    return B_cum


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
            k = quantize_ste_absmax(k, self.kv_cache_bits)
            v = quantize_ste_absmax(v, self.kv_cache_bits)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            q = torch.ao.nn.functional.rope(q, dim=-1, theta=10000.0)
            k = torch.ao.nn.functional.rope(k, dim=-1, theta=10000.0)

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

    def _forward_sequential(self, q, k, v, gate_f, gate_i=None):
        B, H, T, D = q.shape
        S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        ys = []
        for t in range(T):
            k_t = k[:, :, t, :].unsqueeze(-1)
            v_t = v[:, :, t, :].unsqueeze(-2)
            f = gate_f[:, t].unsqueeze(-1).unsqueeze(-1)
            if gate_i is not None:
                i = gate_i[:, t].unsqueeze(-1).unsqueeze(-1)
            else:
                i = 1 - f
            S = f * S + i * (k_t @ v_t)
            q_t = q[:, :, t, :].unsqueeze(-1)
            y = (q_t.transpose(-2, -1) @ S).squeeze(-2)
            ys.append(y)
        return torch.stack(ys, dim=2)