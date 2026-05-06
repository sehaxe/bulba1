import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from bulba1.model.bit_linear import make_linear, activation_quant_ste, QuantizedKVCache


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
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos[:, :, :seq_len, : x.shape[-1]]
        sin = self.sin[:, :, :seq_len, : x.shape[-1]]
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
        self.lambda_init = getattr(cfg, "lambda_init", 0.8)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))

        if cfg.use_mla:
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

        self.rope = RoPE(self.head_dim, cfg.max_ctx_len, cfg.rope_theta)
        self.register_buffer("lambda_val", torch.tensor(self.lambda_init))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, T)
        k = self.rope(k, T)

        use_qkv = getattr(self.cfg, "use_quantized_kv_cache", False)
        if past_kv is not None:
            if use_qkv and isinstance(past_kv, QuantizedKVCache):
                past_kv.append(k, v)
                k, v = past_kv.get()
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
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=(mask is None and T == T_total)
        )

        q2 = q * (0.5**0.5)
        k2 = k * (0.5**0.5)
        out2 = F.scaled_dot_product_attention(
            q2, k2, v, attn_mask=mask, dropout_p=0.0, is_causal=(mask is None and T == T_total)
        )

        lambda_1 = torch.sum(self.lambda_q1 * self.lambda_k1)
        lambda_2 = torch.sum(self.lambda_q2 * self.lambda_k2)
        lambda_val = torch.sigmoid(lambda_1) - torch.sigmoid(lambda_2) + self.lambda_val

        out = out1 - lambda_val * out2

        if self.cfg.use_per_head_gating:
            out = out * self.head_gates.view(1, H, 1, 1)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        if getattr(self.cfg, "use_bitnet_a48", False):
            topk_sparsity = getattr(self.cfg, "a48_attn_topk_sparsity", 0.5)
            if topk_sparsity > 0 and topk_sparsity < 1.0:
                out = activation_quant_ste(out, getattr(self.cfg, "bitnet_activation_bits", 8))
                mask = self._compute_topk_mask(out, topk_sparsity)
                out = out * mask

        out = self.o_proj(out)

        if use_qkv:
            cache = QuantizedKVCache(num_bits=getattr(self.cfg, "kv_cache_bits", 3))
            cache.quantize(k, v)
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
