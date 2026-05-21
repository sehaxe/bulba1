"""
Transformer Block with Attention Residuals, MoE, Mamba, and KDA support.
Optimized for torch.compile with minimal Python overhead.
"""
import torch
import torch.nn as nn
from bulba1.model.bit_linear import topk_sparsify
from bulba1.model.diff_attn import DiffAttention, RMSNorm
from bulba1.model.kda import KimiDeltaAttention
from bulba1.model.kimi_linear import KimiLinear
from bulba1.model.attn_res import AttentionResidual
from bulba1.model.moe import MoELayer

class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.res_scale = nn.Parameter(torch.zeros(1))

        # Determine block type
        pattern = getattr(cfg, "alternating_pattern", None)
        if pattern is not None and layer_idx < len(pattern):
            self.is_attn_block = pattern[layer_idx] == "attn"
        else:
            attn_every = getattr(cfg, "attn_every_n_layers", 4) or 4
            self.is_attn_block = (layer_idx % attn_every) == 0

        # Pre-compute flags to avoid hasattr checks in forward
        self.has_moe = bool(cfg.use_moe) and self.is_attn_block
        self.has_mamba = bool(getattr(cfg, "use_mamba", False)) and not self.is_attn_block

        # BitNet a4.8 settings
        self.use_bitnet_a48 = getattr(cfg, "use_bitnet_a48", False)
        self.a48_topk = getattr(cfg, "a48_attn_topk_sparsity", 0.5) if self.use_bitnet_a48 else 0.0

        # Stochastic depth
        self.sd_prob = getattr(cfg, "stochastic_depth_prob", 0.0)

        # NASA Rule #7: Pre-compute flags to avoid hasattr checks in forward
        self.use_rex = cfg.use_rex and self.has_moe

        # Attention Residuals
        self.use_attn_res = getattr(cfg, "use_attn_res", True)
        self.attn_res_mode = getattr(cfg, "attn_res_mode", "block")

        # Build sublayers
        self.norm1 = RMSNorm(cfg.d_model)
        if self.is_attn_block:
            if getattr(cfg, "use_kda", False):
                self.attn = KimiDeltaAttention(cfg)
            else:
                self.attn = DiffAttention(cfg)
            self.norm2 = RMSNorm(cfg.d_model)
            if self.has_moe:
                self.moe = MoELayer(cfg, layer_idx)
            if self.use_attn_res:
                self.attn_res = AttentionResidual(cfg.d_model, getattr(cfg, "attn_res_recency_bias_init", 10.0))
                self.mlp_res = AttentionResidual(cfg.d_model, getattr(cfg, "attn_res_recency_bias_init", 10.0))
        else:
            if self.has_mamba:
                pass  # MambaBlock disabled in current config
            if self.use_attn_res:
                self.attn_res = AttentionResidual(cfg.d_model, getattr(cfg, "attn_res_recency_bias_init", 10.0))

    def _get_zero_aux(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-allocated zero tensor for aux loss (avoids CPU overhead)."""
        return x.new_zeros(())

    def _route_history(self, blocks_history: tuple[torch.Tensor, ...], mod_idx: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Vectorized MoD routing for block history."""
        if mod_idx is None:
            return blocks_history
        # Stack history: (B, T, num_blocks, D) -> single GPU gather
        stacked = torch.stack(blocks_history, dim=2)
        idx_exp = mod_idx.detach().unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, stacked.size(2), stacked.size(-1)
        )
        routed = stacked.gather(1, idx_exp)  # (B, n_keep, num_blocks, D)
        return tuple(routed.unbind(dim=2))

    def forward(
        self,
        x: torch.Tensor,
        prev_experts=None,
        past_kv=None,
        blocks_history: tuple[torch.Tensor, ...] | None = None,
        mod_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Stochastic depth (training only)
        if self.training and self.sd_prob > 0.0 and torch.rand(1, device=x.device) < self.sd_prob:
            return x, self._get_zero_aux(x), None

        h = x
        new_kv = None
        total_aux = self._get_zero_aux(x)

        # Route history for MoD if needed
        blocks_routed = self._route_history(blocks_history, mod_idx) if self.use_attn_res and blocks_history else None

        if self.is_attn_block:
            # ── Attention sublayer ──
            if blocks_routed is not None:
                h_attn = self.attn_res(list(blocks_routed), h)
            else:
                h_attn = h
            attn_out, new_kv, attn_z = self.attn(self.norm1(h_attn), past_kv=past_kv)
            if self.use_bitnet_a48:
                attn_out = topk_sparsify(attn_out, self.a48_topk)
            h = h + attn_out
            attn_z_coef = getattr(self.cfg, "attn_z_loss_coef", 0.0001)
            if attn_z_coef > 0:
                total_aux = total_aux + attn_z * attn_z_coef

            # ── MoE sublayer ──
            if blocks_routed is not None:
                h_mlp = self.mlp_res(list(blocks_routed), h)
            else:
                h_mlp = h
            if self.has_moe:
                moe_out, aux_loss = self.moe(self.norm2(h_mlp), prev_experts)
                h = h + moe_out
                total_aux = total_aux + aux_loss
        else:
            # ── Mamba sublayer (fallback to standard if disabled) ──
            if blocks_routed is not None:
                h_mamba = self.attn_res(list(blocks_routed), h)
            else:
                h_mamba = h
            # Mamba logic omitted as use_mamba=False in night_smart config

        # Residual scaling (tanh-bounded for stability)
        h = x + (h - x) * self.res_scale.tanh()
        return h, total_aux, new_kv
