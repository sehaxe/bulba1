"""
Triton-accelerated BitLinear kernel.
Fused: weight quantization (STE) + matrix multiplication.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_kernel(w_ptr, scale_ptr, bits: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """Quantize weights using absmax STE."""
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    w_vals = tl.load(w_ptr + pid * N + offs).to(tl.float32)
    scale = tl.max(tl.abs(w_vals)) + 1e-8
    w_quant = tl.round(w_vals / scale)
    w_quant = tl.clamp(w_quant, -1.0, 1.0)
    # Store quantized weight
    tl.store(w_ptr + pid * N + offs, w_quant)
    tl.store(scale_ptr + pid, scale)


@triton.jit
def _bitlinear_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    stride_xm, stride_xn, stride_wn, stride_wn2,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Fused BitLinear: quantize weights + matmul."""
    pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = min(num_pid_in_group, M * N - group_id * num_pid_in_group)
    pid_m = first_pid_m + (pid % group_size) // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_xm = offs_m[:, None]
    offs_xn = offs_n[None, :]
    offs_wn = offs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load and quantize weight once for this block
    scale = tl.load(scale_ptr + pid_n).to(tl.float32)
    w_vals = tl.load(w_ptr + pid_n * N + offs_n).to(tl.float32)
    w_quant = tl.round(w_vals / scale)
    w_quant = tl.clamp(w_quant, -1.0, 1.0)

    for k in range(0, M):
        x = tl.load(x_ptr + offs_m * stride_xm + k * stride_xn + offs_xn).to(tl.float32)
        acc += x * w_quant

    out = acc * scale
    tl.store(out_ptr + offs_m * stride_om + offs_n * stride_on, out.to(tl.float16))


def triton_bitlinear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """Triton-accelerated BitLinear forward pass."""
    B, M, N = x.shape[0], x.shape[1], weight.shape[0]
    assert weight.shape[1] == N, f"Shape mismatch: {weight.shape} vs ({B}, {M}, {N})"

    out = torch.zeros(B, M, weight.shape[0], device=x.device, dtype=torch.float16)
    BLOCK_M = 16
    BLOCK_N = 64

    grid = (M * weight.shape[0],)
    _bitlinear_kernel[grid](
        x, weight, None, out,
        B, M, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N
    )

    if bias is not None:
        out = out + bias

    return out


def triton_bitlinear_available():
    """Check if Triton BitLinear can be used."""
    if not is_triton_available():
        return False
    if not torch.cuda.is_available():
        return False
    return True