import torch
import torch.nn as nn
import torch.nn.functional as F


class MHC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.iterations = cfg.mhc_iterations
        self.rescale = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def sinkhorn_knopp(self, mat, n_iter=5):
        mat = F.softmax(mat, dim=-1)
        for _ in range(n_iter):
            mat = mat / (mat.sum(dim=-2, keepdim=True) + 1e-8)
            mat = mat / (mat.sum(dim=-1, keepdim=True) + 1e-8)
        return mat

    def forward(self, x):
        B, T, D = x.shape
        mat = self.rescale(x)
        mat = mat.view(B * T, D, 1)
        perm = self.sinkhorn_knopp(mat, self.iterations)
        out = (perm * mat).sum(dim=-1).view(B, T, D)
        return out
