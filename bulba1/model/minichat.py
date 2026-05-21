"""
MiniChat Model - Core architecture with MoE, Mamba, KDA, MoD, and MTP.
Optimized for memory efficiency and fast inference.
"""
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
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)

        self.num_blocks = getattr(cfg, "num_unique_blocks", self.cfg.n_layers)
        self.repeats = getattr(cfg, "recurrent_repeats", 1)
        self.merge_every = getattr(cfg, "merge_every_n_layers", 2)
        self.use_mod = getattr(cfg, "use_mixture_of_depths", False)

        self._inference_merge_ratio = getattr(cfg, "inference_merge_ratio", 0.3)
        self.merger = None
        if self._inference_merge_ratio > 0 and not getattr(cfg, "use_attn_res", True):
            self._train_merger = TokenMerger(self.cfg.d_model, self._inference_merge_ratio)

        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(self.num_blocks)])
        self.norm = RMSNorm(self.cfg.d_model)

        if getattr(cfg, "tied_embeddings", True):
            self.lm_head = TiedHead(self.embedding.weight)
        else:
            self.lm_head = (
                make_linear(cfg, self.cfg.d_model, self.cfg.vocab_size, bias=False, quantize_input=False)
                if getattr(cfg, "bitlinear_lm_head", False)
                else nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
            )

        if self.use_mod:
            self.mod_gates = nn.ModuleList([
                MoDGate(self.cfg.d_model, getattr(cfg, "mod_capacity", 0.75))
                for _ in range(self.num_blocks)
            ])

        self.use_mtp = getattr(cfg, "use_mtp", False)
        self.num_mtp_heads = getattr(cfg, "num_mtp_heads", 0)
        if self.use_mtp and self.num_mtp_heads > 0:
            self.mtp_norm = RMSNorm(self.cfg.d_model)
            
            def _make_mtp_layer(out_features: int, tied: bool = False) -> nn.Module:
                if tied:
                    return TiedHead(self.embedding.weight)
                return make_linear(cfg, self.cfg.d_model, out_features, bias=False, quantize_input=False)
            
            self.mtp_projections = nn.ModuleList([
                _make_mtp_layer(self.cfg.d_model) for _ in range(self.num_mtp_heads - 1)
            ])
            self.mtp_heads = nn.ModuleList([
                _make_mtp_layer(self.cfg.vocab_size, tied=cfg.tied_embeddings)
                for _ in range(self.num_mtp_heads)
            ])

        self.num_clr_tokens = getattr(cfg, "num_clr_tokens", 0)
        if self.num_clr_tokens > 0:
            self.clr_tokens = nn.Parameter(torch.randn(1, self.num_clr_tokens, self.cfg.d_model))
            
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
                std = d_model ** -0.5 if fan_in == d_model else fan_in ** -0.5
            elif getattr(self.cfg, "use_bitlinear", False) and getattr(self.cfg, "bitnet_init_std", 0.001) > 0:
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
        
        if self.num_clr_tokens > 0:
            clr = self.clr_tokens.expand(B, -1, -1)
            x = torch.cat([clr, x], dim=1)

        total_aux_loss = x.new_zeros(())
        prev_experts = None
        use_ckpt = self.cfg.use_gradient_checkpointing
        ckpt_every = getattr(self.cfg, "checkpoint_every_n_layers", 1) or 1
        eff_hidden = self.cfg.d_model
        merge_idx = None

        use_attn_res = getattr(self.cfg, "use_attn_res", True)
        if use_attn_res:
            # ИСПРАВЛЕНО: используем кортеж (tuple) вместо списка.
            # Кортежи иммутабельны, поэтому torch.utils.checkpoint и autograd не ломаются.
            blocks_history = (x,)
            total_layers = self.repeats * self.num_blocks
            num_blocks_cfg = getattr(self.cfg, "attn_res_num_blocks", 8)
            layers_per_block = max(1, (total_layers + num_blocks_cfg - 1) // num_blocks_cfg)
        else:
            blocks_history = None

        for repeat in range(self.repeats):
            for i, block in enumerate(self.blocks):
                effective_layer = repeat * self.num_blocks + i

                if self.use_mod:
                    gate = self.mod_gates[i]
                    x_routed, mod_idx = gate(x)
                    if x_routed.size(1) > 0:
                        if use_ckpt and effective_layer > 0 and effective_layer % ckpt_every == 0:
                            mod_out, aux, _ = ckpt(block, x_routed, prev_experts, None, blocks_history, mod_idx, use_reentrant=False)
                        else:
                            mod_out, aux, _ = block(x_routed, prev_experts, None, blocks_history, mod_idx)
                        
                        idx_expanded = mod_idx.detach().unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        x = x.scatter(1, idx_expanded, mod_out.to(x.dtype))
                    else:
                        aux = x.new_zeros(())
                elif use_ckpt and effective_layer > 0 and effective_layer % ckpt_every == 0:
                    x, aux, _ = ckpt(block, x, prev_experts, None, blocks_history, None, use_reentrant=False)
                else:
                    x, aux, _ = block(x, prev_experts, None, blocks_history, None)

                total_aux_loss = total_aux_loss + aux
                
                if self.cfg.use_rex and self.cfg.use_moe and getattr(block, 'has_moe', False):
                    prev_experts = block.moe.get_expert_modules()

                if use_attn_res:
                    attn_res_mode = getattr(self.cfg, "attn_res_mode", "block")
                    if attn_res_mode == "full" or (effective_layer + 1) % layers_per_block == 0:
                        # ИСПРАВЛЕНО: создаём новый кортеж вместо in-place .append()
                        blocks_history = blocks_history + (x,)

                if self.merger is not None and (effective_layer + 1) % self.merge_every == 0:
                    x, merge_idx = self.merger(x)

        if self.merger is not None and merge_idx is not None:
            x = self.merger.unmerge(x, merge_idx, (B, T + self.num_clr_tokens, eff_hidden))

        x = self.norm(x)
        logits = self.lm_head(x)

        mtp1, mtp2 = None, None
        if self.use_mtp and self.num_mtp_heads > 0:
            h_mtp = self.mtp_norm(x)
            mtp1 = self.mtp_heads[0](h_mtp)
            if self.num_mtp_heads > 1:
                h_mtp = F.silu(self.mtp_projections[0](h_mtp))
                mtp2 = self.mtp_heads[1](h_mtp)

        return logits, mtp1, mtp2, total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids.clone()
        max_ctx = getattr(self.cfg, "max_ctx_len", 2048)
        
        for _ in range(max_new_tokens):
            inp = ids[:, -max_ctx:]
            logits, _, _, _ = self(inp)
            next_logits = logits[:, -1, :]
            
            if temperature > 0:
                next_logits = next_logits / temperature
            
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_logits, k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)
            else:
                if temperature <= 1e-5:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            
            ids = torch.cat([ids, next_token], dim=1)
        return ids
