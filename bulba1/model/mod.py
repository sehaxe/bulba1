import torch
import torch.nn as nn


class MoDGate(nn.Module):
    def __init__(self, d_model: int, capacity: float = 0.75):
        super().__init__()
        self.capacity = capacity
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        scores = self.proj(x).squeeze(-1)
        n_keep = max(1, int(T * self.capacity))
        _, top_idx = scores.topk(n_keep, dim=-1, sorted=False)
        mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        routed = x[mask].view(B, n_keep, D)
        return routed, top_idx  # Return indices instead of mask
