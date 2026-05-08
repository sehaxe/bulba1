import torch.nn as nn
from mamba_ssm import Mamba as Mamba3

class MambaBlock(nn.Module):
    """Блок Mamba-3 (требуется mamba-ssm >= 2.3.0)."""
    def __init__(self, cfg):
        super().__init__()
        self.mamba = Mamba3(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )
    def forward(self, x):
        return self.mamba(x)