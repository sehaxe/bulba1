import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from bulba1.model.bit_linear import make_linear
from bulba1.model.block import Block
from bulba1.model.diff_attn import RMSNorm
from bulba1.model.token_merging import TokenMerger
from bulba1.model.mod import MoDGate


class TiedHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        return F.linear(x, self.weight)


class MiniChat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.num_blocks = getattr(cfg, "num_unique_blocks", cfg.n_layers)
        self.repeats = getattr(cfg, "recurrent_repeats", 1)
        self.merge_every = getattr(cfg, "merge_every_n_layers", 2)
        self.use_mod = getattr(cfg, "use_mixture_of_depths", False)

        self._inference_merge_ratio = getattr(cfg, "inference_merge_ratio", 0.3)
        self.merger = None
        if self._inference_merge_ratio > 0:
            self._train_merger = TokenMerger(cfg.d_model, self._inference_merge_ratio)

        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(self.num_blocks)])
        self.norm = RMSNorm(cfg.d_model)

        if getattr(cfg, "tied_embeddings", True):
            self.lm_head = TiedHead(self.embedding.weight)
        else:
            self.lm_head = (
                make_linear(cfg, cfg.d_model, cfg.vocab_size, bias=False, quantize_input=False)
                if getattr(cfg, "bitlinear_lm_head", False)
                else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            )

        if self.use_mod:
            self.mod_gates = nn.ModuleList([
                MoDGate(cfg.d_model, getattr(cfg, "mod_capacity", 0.75))
                for _ in range(self.num_blocks)
            ])

        if cfg.use_mtp:
            self.mtp_norm = RMSNorm(cfg.d_model)
            use_bit_mtp = getattr(cfg, "bitlinear_mtp", False)
            tied = getattr(cfg, "tied_embeddings", True)
            self.mtp_projections = nn.ModuleList([
                make_linear(cfg, cfg.d_model, cfg.d_model, bias=False, quantize_input=False)
                if use_bit_mtp else nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                for _ in range(cfg.num_mtp_heads)
            ])
            if tied:
                w = self.embedding.weight
                self.mtp_heads = nn.ModuleList([TiedHead(w) for _ in range(cfg.num_mtp_heads)])
            else:
                self.mtp_heads = nn.ModuleList([
                    make_linear(cfg, cfg.d_model, cfg.vocab_size, bias=False, quantize_input=False)
                    if use_bit_mtp else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                    for _ in range(cfg.num_mtp_heads)
                ])

        self.clr_tokens = nn.Parameter(torch.randn(1, cfg.num_clr_tokens, cfg.d_model))
        self.apply(self._init_weights)

    def train(self, mode=True):
        super().train(mode)
        self.merger = None
        return self

    def eval(self):
        super().eval()
        if hasattr(self, "_train_merger") and self._train_merger is not None:
            self.merger = self._train_merger
        return self

    def _init_weights(self, module):
        init_std = getattr(self.cfg, "init_std", 0.02) or 0.02
        use_mup = getattr(self.cfg, "use_mup_init", True)
        d_model = self.cfg.d_model

        if isinstance(module, nn.Linear):
            if use_mup:
                fan_in = module.weight.size(1)
                if fan_in == d_model:
                    std = d_model ** -0.5
                else:
                    std = fan_in ** -0.5
            elif (getattr(self.cfg, "use_bitlinear", False)
                  and getattr(self.cfg, "bitnet_init_std", 0.001) > 0):
                std = getattr(self.cfg, "bitnet_init_std", 0.001)
            elif getattr(self.cfg, "depth_scaled_init", False):
                std = init_std / math.sqrt(2 * (getattr(self.cfg, "n_layers", 1) or 1))
            else:
                std = init_std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = d_model ** -0.5 if use_mup else init_std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        if self.cfg.num_clr_tokens > 0:
            clr = self.clr_tokens.expand(B, -1, -1)
            x = torch.cat([clr, x], dim=1)

        total_aux_loss = 0.0
        prev_experts = None
        use_ckpt = self.cfg.use_gradient_checkpointing
        ckpt_every = getattr(self.cfg, "checkpoint_every_n_layers", 1) or 1
        eff_hidden = self.cfg.d_model
        merge_idx = None

        for repeat in range(self.repeats):
            for i, block in enumerate(self.blocks):
                effective_layer = repeat * self.num_blocks + i

                if self.use_mod:
                    gate = self.mod_gates[i]
                    x_routed, mod_idx = gate(x)  # mod_idx: (B, n_keep)
                    if x_routed.size(1) > 0:
                        if use_ckpt and effective_layer > 0 and effective_layer % ckpt_every == 0:
                            mod_out, aux, _ = ckpt(block, x_routed, prev_experts, use_reentrant=False)
                        else:
                            mod_out, aux, _ = block(x_routed, prev_experts)
                        mod_out = mod_out.to(x.dtype)
                        # Scatter outputs back - detach indices to avoid grad issues
                        x_new = x.clone()
                        idx_expanded = mod_idx.detach().unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x_new = x_new.scatter(1, idx_expanded, mod_out)
                        x = x_new
                    else:
                        aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                elif use_ckpt and effective_layer > 0 and effective_layer % ckpt_every == 0:
                    x, aux, _ = ckpt(block, x, prev_experts, use_reentrant=False)
                else:
                    x, aux, _ = block(x, prev_experts)

                total_aux_loss += aux
                if self.cfg.use_rex and self.cfg.use_moe and hasattr(block, "moe"):
                    prev_experts = block.moe.get_expert_modules()

                if self.merger is not None and (effective_layer + 1) % self.merge_every == 0:
                    x, merge_idx = self.merger(x)

        if self.merger is not None and merge_idx is not None:
            x = self.merger.unmerge(x, merge_idx, (B, T + self.cfg.num_clr_tokens, eff_hidden))

        x = self.norm(x)
        logits = self.lm_head(x)

        if self.cfg.use_mtp:
            mtp_logits = []
            h_mtp = self.mtp_norm(x)
            for i in range(self.cfg.num_mtp_heads):
                mtp_logits.append(self.mtp_heads[i](h_mtp))
                if i < self.cfg.num_mtp_heads - 1:
                    h_mtp = F.silu(self.mtp_projections[i](h_mtp))
        else:
            mtp_logits = [None, None, None]

        return logits, mtp_logits[0], mtp_logits[1], total_aux_loss


