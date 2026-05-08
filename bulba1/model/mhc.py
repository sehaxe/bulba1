import torch
import torch.nn as nn

class MHC(nn.Module):
    """DeepSeek Manifold-Constrained Hyper-Connection (Cayley transform)."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def cayley(self, A):
        I = torch.eye(self.d_model, device=A.device).unsqueeze(0).unsqueeze(0)
        C = torch.linalg.solve(I + A, I - A)
        return C

    def forward(self, x):
        B, T, D = x.shape
        A = self.proj(x)                               # (B, T, D)
        # Create skew-symmetric matrix of shape (B, T, D, D)
        A_skew = A.unsqueeze(-1) - A.unsqueeze(-2)     # (B, T, D, D), a_ij = A_i - A_j
        M = self.cayley(A_skew)                        # orthogonal matrix
        return torch.matmul(x.unsqueeze(-2), M).squeeze(-2)