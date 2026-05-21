import torch
import torch.nn as nn
from bulba1.model.diff_attn import RMSNorm

class AttentionResidual(nn.Module):
    """
    Attention Residuals (AttnRes) - Block variant.
    """
    def __init__(self, d_model: int, bias_init: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.norm = RMSNorm(d_model)
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(
        self,
        blocks: tuple[torch.Tensor, ...] | list[torch.Tensor],
        partial_block: torch.Tensor
    ) -> torch.Tensor:
        V = torch.stack(list(blocks) + [partial_block], dim=0)
        K = self.norm(V)
        query = self.proj.weight.view(-1)
        
        logits = torch.einsum("d, n b t d -> n b t", query, K)
        logits[-1] = logits[-1] + self.bias
        
        # FIX: Upcast to float32 for stable softmax, then cast back
        weights = logits.float().softmax(dim=0).to(V.dtype)
        
        return torch.einsum("n b t, n b t d -> b t d", weights, V)