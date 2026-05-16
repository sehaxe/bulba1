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


def q_int4_ste(x: torch.Tensor) -> torch.Tensor:
    """4-bit integer quantization (absmean) for activations."""
    beta = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = beta / 7.0
    x_norm = x / scale
    x_quant = torch.round(torch.clamp(x_norm, -8, 7))
    return ((x_quant - x_norm).detach() + x_norm) * scale


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
        use_triton = getattr(cfg, "use_triton_bitlinear", False)
        return BitLinear(
            in_features,
            out_features,
            bias=bias,
            activation_bits=abits,
            quantize_input=quantize_input,
            use_triton=use_triton,
        )
    return nn.Linear(in_features, out_features, bias=bias)


# Future: Triton-based BitLinear for acceleration
# Set use_triton_bitlinear=True in config to enable (when available)
def triton_bitlinear_available():
    """Check if Triton BitLinear kernel is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
        quantize_input: bool = True,
        use_triton: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.quantize_input = quantize_input
        self._use_triton = use_triton
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
        # Try Triton if enabled and available
        if (getattr(self, '_use_triton', False) and 
            self.training and 
            x.is_cuda and 
            self.weight.is_cuda):
            try:
                from bulba1.triton_ops.bitlinear import triton_bitlinear_available
                if triton_bitlinear_available():
                    from bulba1.triton_ops.bitlinear import triton_bitlinear_forward
                    return triton_bitlinear_forward(x, self.weight, self.bias)
            except Exception:
                pass  # Fall back to Python
        
        # Python implementation (default)
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


