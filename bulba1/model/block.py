import torch
import torch.nn as nn
from bulba1.model.diff_attn import DiffAttention, RMSNorm
from bulba1.model.moe import MoELayer
from bulba1.model.mamba import Mamba2SSD
from bulba1.model.kda import KimiDeltaAttention


class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        pattern = getattr(cfg, "alternating_pattern", None)
        if pattern is not None and layer_idx < len(pattern):
            self.is_attn_block = pattern[layer_idx] == "attn"
        else:
            attn_every = getattr(cfg, "attn_every_n_layers", 4)
            self.is_attn_block = (layer_idx % attn_every) == 0

        if self.is_attn_block:
            self.norm1 = RMSNorm(cfg.d_model)
            if getattr(cfg, "use_kda", False):
                self.attn = KimiDeltaAttention(cfg)
            else:
                self.attn = DiffAttention(cfg)
            self.norm2 = RMSNorm(cfg.d_model)
            if cfg.use_moe:
                self.moe = MoELayer(cfg, layer_idx)
        else:
            self.norm1 = RMSNorm(cfg.d_model)
            self.mamba = Mamba2SSD(cfg)

    def forward(self, x: torch.Tensor, prev_experts=None, past_kv=None):
        h = x

        if self.is_attn_block:
            attn_out, new_kv, attn_z = self.attn(self.norm1(h), past_kv=past_kv)
            h = h + attn_out

            if self.cfg.use_moe:
                moe_out, aux_loss = self.moe(self.norm2(h), prev_experts)
            else:
                moe_out = torch.zeros_like(h)
                aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            h = h + moe_out
            total_aux = aux_loss + attn_z * getattr(self.cfg, "attn_z_loss_coef", 0.0001)
        else:
            h = h + self.mamba(self.norm1(h))
            total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            new_kv = None

        return h, total_aux, new_kv
