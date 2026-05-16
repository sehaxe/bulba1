import torch
import torch.nn as nn


class TokenMerger(nn.Module):
    def __init__(self, d_model: int, ratio: float = 0.3):
        super().__init__()
        self.ratio = ratio
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        n_keep = max(1, int(T * (1.0 - self.ratio)))
        if n_keep >= T:
            return x, None
        scores = self.score(x).squeeze(-1)
        _, idx = scores.topk(n_keep, dim=-1, sorted=False)
        keep = x.gather(1, idx[:, :, None].expand(-1, -1, D))
        return keep, idx

    def unmerge(self, keep: torch.Tensor, idx: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        if idx is None:
            return keep
        B, T, D = orig_shape
        out = keep.new_zeros(B, T, D)
        out.scatter_(1, idx[:, :, None].expand(-1, -1, D), keep)
        return out
