"""
Rotary Positional Embedding (RoPE) implementation.

Provides efficient rotary position embeddings for transformer models.
"""

import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """Rotary Positional Embedding layer."""
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
        cos = self.cos[:, :, :seq_len, : x.shape[-1]]
        sin = self.sin[:, :, :seq_len, : x.shape[-1]]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin