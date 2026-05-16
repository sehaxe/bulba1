"""
Kimi Delta Attention (KDA) module.
Efficient sequential implementation with optional torch.compile support.
"""

import torch
import torch.nn as nn

from bulba1.model.bit_linear import make_linear
from bulba1.model.rope import RoPE


class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.gate_dim = getattr(cfg, "kda_gate_dim", 16)
        self.use_compile = getattr(cfg, "compile", False) and hasattr(torch, "compile")
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

        if self.use_rope:
            self.rope = RoPE(self.head_dim, getattr(cfg, "max_ctx_len", 4096), getattr(cfg, "rope_theta", 10000.0))

    def _forward_sequential(self, q, k, v, gate_f, gate_i=None):
        """Sequential scan implementation - reliable and correct."""
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
            q = self.rope(q, T)
            k = self.rope(k, T)

        gate_logits = self.gate_proj(x).view(B, T, H, -1)
        if self.use_double_gate:
            gate = torch.sigmoid(self.gate_out(gate_logits))
            gate_f = gate[..., 0].transpose(1, 2)
            gate_i = gate[..., 1].transpose(1, 2)
        else:
            gate = torch.sigmoid(self.gate_out(gate_logits).squeeze(-1))
            gate_f = gate.transpose(1, 2)
            gate_i = None

        if self.use_compile and self.training:
            out = torch.compile(self._forward_sequential)(q, k, v, gate_f, gate_i)
        else:
            out = self._forward_sequential(q, k, v, gate_f, gate_i)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), None, torch.tensor(0.0, device=x.device)


