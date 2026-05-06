import os
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor selective_scan_fwd(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void selective_scan_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    float* __restrict__ out,
    int batch, int seq_len, int d_inner, int d_state
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= batch || d >= d_inner) return;

    int stride = seq_len * d_inner;
    int b_stride = seq_len * d_state;

    for (int s = tid; s < d_state; s += blockDim.x) {
        float h = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            int idx = b * stride + t * d_inner + d;
            int b_idx = b * b_stride + t * d_state + s;
            float dt_val = dt[idx];
            float x_val = x[idx];
            float a_val = A[d * d_state + s];
            float b_val = B[b_idx];
            float c_val = C[b_idx];

            float da = expf(a_val * dt_val);
            float db = dt_val * b_val * x_val;
            h = da * h + db;
            out[idx] += h * c_val;
        }
    }

    if (tid == 0) {
        for (int t = 0; t < seq_len; t++) {
            int idx = b * stride + t * d_inner + d;
            out[idx] += D[d] * x[idx];
        }
    }
}

torch::Tensor selective_scan_fwd(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
) {
    int batch = x.size(0);
    int seq_len = x.size(1);
    int d_inner = x.size(2);
    int d_state = A.size(1);

    auto out = torch::zeros_like(x);

    dim3 blocks(batch, d_inner);
    int threads = 256;

    selective_scan_fwd_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        dt.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, seq_len, d_inner, d_state
    );

    return out;
}
"""

try:
    _selective_scan_cuda = load_inline(
        name="selective_scan_cuda_v2",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["selective_scan_fwd"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++14"],
        extra_cflags=["-std=c++14"],
        verbose=False,
    )
    HAS_CUDA_KERNEL = True
except Exception as e:
    print(f"CUDA kernel compilation failed: {e}")
    _selective_scan_cuda = None
    HAS_CUDA_KERNEL = False


def selective_scan_cuda(x, dt, A, B, C, D):
    if not HAS_CUDA_KERNEL or not x.is_cuda:
        return None
    return _selective_scan_cuda.selective_scan_fwd(x, dt, A, B, C, D)
