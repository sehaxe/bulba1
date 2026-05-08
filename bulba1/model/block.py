import torch
import torch.nn as nn
from bulba1.model.diff_attn import DiffAttention, RMSNorm
from bulba1.model.moe import MoELayer
from bulba1.model.mamba import MambaBlock
from bulba1.model.kda import KimiDeltaAttention
from bulba1.model.bit_linear import q_int4_v2, hadamard_transform

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
            if cfg.use_mamba:
                self.mamba = MambaBlock(cfg)

        self.use_mhc = getattr(cfg, "use_mhc", True)
        if self.use_mhc:
            from bulba1.model.mhc import MHC
            self.mhc = MHC(cfg.d_model)
        else:
            self.mhc = None

        self.use_bitnet_a48 = getattr(cfg, "use_bitnet_a48", True)
        if self.use_bitnet_a48:
            self.a48_topk = getattr(cfg, "a48_attn_topk_sparsity", 0.5)

    def forward(self, x: torch.Tensor, prev_experts=None, past_kv=None):
        h = x

        if self.use_bitnet_a48:
            h = q_int4_v2(hadamard_transform(h))

        if self.is_attn_block:
            attn_out, new_kv, attn_z = self.attn(self.norm1(h), past_kv=past_kv)
            if self.use_bitnet_a48:
                attn_out = topk_sparsify(attn_out, self.a48_topk)
            h = h + attn_out

            if self.cfg.use_moe:
                moe_out, aux_loss = self.moe(self.norm2(h), prev_experts)
            else:
                moe_out = torch.zeros_like(h)
                aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            h = h + moe_out
            total_aux = aux_loss + attn_z * getattr(self.cfg, "attn_z_loss_coef", 0.0001)
        else:
            if hasattr(self, 'mamba'):
                mamba_out = self.mamba(self.norm1(h))
            else:
                mamba_out = torch.zeros_like(h)
            if self.use_bitnet_a48:
                mamba_out = topk_sparsify(mamba_out, self.a48_topk)
            h = h + mamba_out
            total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            new_kv = None

        if self.mhc is not None:
            h = self.mhc(h)

        return h, total_aux, new_kv