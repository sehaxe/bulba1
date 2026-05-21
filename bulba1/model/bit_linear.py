import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_b158(w: torch.Tensor) -> torch.Tensor:
    """1.58-bit weights: Sign(w - mean(w)) * mean(|w|)"""
    gamma = w.abs().mean().clamp_min(1e-8)
    w_norm = w / gamma
    w_quant = torch.round(torch.clamp(w_norm, -1.0, 1.0))
    return ((w_quant - w_norm).detach() + w_norm) * gamma

def activation_quant_ste(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """8-bit absmax for attention/FFN inputs (Paper 2411.04965 Eq.3)"""
    Q_b = 2 ** (num_bits - 1)
    gamma = x.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8)
    scale = gamma / (Q_b - 1)
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -(Q_b-1), Q_b-1))
    return ((x_quant - x_norm).detach() + x_norm) * scale

def activation_quant_ste_absmean(x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """4-bit absmean for intermediate states (Paper 2411.04965 Eq.7)"""
    Q_b = 2 ** (num_bits - 1)
    beta = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = beta / Q_b
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -Q_b, Q_b-1))
    return ((x_quant - x_norm).detach() + x_norm) * scale

def quantize_ste_absmax(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Backwards-compatible alias for activation_quant_ste."""
    return activation_quant_ste(x, num_bits)

def topk_sparsify(x: torch.Tensor, k: float = 0.5) -> torch.Tensor:
    """Top-K sparsification for down-proj/out-attn (BitNet a4.8 §2.1 Eq.1)"""
    abs_x = x.abs().float()
    num_keep = max(1, int(int(x.shape[-1]) * k))
    topk_vals, _ = torch.topk(abs_x, num_keep, dim=-1)
    threshold = topk_vals[..., -1:]
    mask = abs_x >= threshold
    return (mask.to(x.dtype) * x - x).detach() + x if x.requires_grad else mask.to(x.dtype) * x

def hadamard_transform(x):
    *batch_dims, D = x.shape
    x = x.reshape(-1, D)
    pad_m = 1
    while pad_m < D: pad_m <<= 1
    if pad_m != D: x = F.pad(x, (0, pad_m - D))
    N = x.size(0)
    x = x[:, None, :]
    block = 1
    while block < pad_m:
        x = x.view(N, -1, 2, block)
        u, v = x[:, :, 0], x[:, :, 1]
        x = torch.stack([u + v, u - v], dim=2)
        block <<= 1
    return (x.view(N, pad_m)[:, :D] / math.sqrt(pad_m)).reshape(*batch_dims, D)

def q_int4_v2(x): return hadamard_transform(x)

class QuantizedKVCache:
    def __init__(self, num_bits: int = 3):
        self.num_bits = num_bits
        self.k_scale = self.v_scale = self.k_quant = self.v_quant = None
    def quantize(self, k: torch.Tensor, v: torch.Tensor):
        Q = 2 ** (self.num_bits - 1)
        self.k_scale = k.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8) / (Q - 1)
        self.v_scale = v.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-8) / (Q - 1)
        self.k_quant = torch.round(torch.clamp(k / self.k_scale, -(Q - 1), Q - 1)).to(torch.int8)
        self.v_quant = torch.round(torch.clamp(v / self.v_scale, -(Q - 1), Q - 1)).to(torch.int8)
    def dequantize(self) -> tuple:
        if self.k_quant is None: return None, None
        return self.k_quant.float() * self.k_scale, self.v_quant.float() * self.v_scale
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k_quant is None: return self.quantize(k_new, v_new)
        k_old, v_old = self.dequantize()
        self.quantize(torch.cat([k_old, k_new], dim=2), torch.cat([v_old, v_new], dim=2))
    def get(self) -> tuple: return self.dequantize()

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False, activation_bits=8, quantize_input=True, use_triton=False, **_):
        super().__init__()
        self.activation_bits, self.quantize_input, self._use_triton = activation_bits, quantize_input, use_triton
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None: nn.init.zeros_(self.bias)
    def forward(self, x):
        # Fallback to Python if Triton fails or is disabled
        if self._use_triton and self.training and x.is_cuda:
            try:
                from bulba1.triton_ops.bitlinear import triton_bitlinear_forward
                return triton_bitlinear_forward(x, self.weight, self.bias)
            except Exception: pass
        w = ste_b158(self.weight).to(x.dtype)
        if self.quantize_input:
            x = activation_quant_ste_absmean(x, 4) if self.activation_bits == 4 else activation_quant_ste(x, 8)
        return F.linear(x, w, self.bias)

def make_linear(cfg, in_f, out_f, bias=False, quantize_input=True):
    if getattr(cfg, "use_bitlinear", False):
        abits = getattr(cfg, "bitnet_activation_bits", 8)
        use_triton = getattr(cfg, "use_triton_bitlinear", False)
        return BitLinear(in_f, out_f, bias=bias, activation_bits=abits, quantize_input=quantize_input, use_triton=use_triton)
    return nn.Linear(in_f, out_f, bias=bias)
