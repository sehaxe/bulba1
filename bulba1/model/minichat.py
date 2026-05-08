import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.block import Block
from bulba1.model.diff_attn import RMSNorm
from bulba1.model.bit_linear import make_linear


class MiniChat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = (
            make_linear(cfg, cfg.d_model, cfg.vocab_size, bias=False, quantize_input=False)
            if getattr(cfg, "bitlinear_lm_head", False)
            else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        )

        if cfg.use_mtp:
            self.mtp_norm = RMSNorm(cfg.d_model)
            use_bit_mtp = getattr(cfg, "bitlinear_mtp", False)
            self.mtp_projections = nn.ModuleList(
                [
                    make_linear(cfg, cfg.d_model, cfg.d_model, bias=False, quantize_input=False)
                    if use_bit_mtp
                    else nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                    for _ in range(cfg.num_mtp_heads)
                ]
            )
            self.mtp_heads = nn.ModuleList(
                [
                    make_linear(cfg, cfg.d_model, cfg.vocab_size, bias=False, quantize_input=False)
                    if use_bit_mtp
                    else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                    for _ in range(cfg.num_mtp_heads)
                ]
            )

        self.register_buffer("clr_tokens", torch.randn(1, cfg.num_clr_tokens, cfg.d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = self.cfg.init_std
            if (
                getattr(self.cfg, "use_bitlinear", False)
                and getattr(self.cfg, "bitnet_init_std", 0.001) > 0
            ):
                std = getattr(self.cfg, "bitnet_init_std", 0.001)
            elif self.cfg.depth_scaled_init:
                std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        checkpoint_every: int = 1,
        past_kvs=None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        x = self.embedding(input_ids)

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
            if self.cfg.num_clr_tokens > 0:
                clr = self.clr_tokens.expand(B, -1, -1)
                x = torch.cat([clr, x], dim=1)

        total_aux_loss = 0.0
        prev_experts = None
        new_past_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if self.cfg.use_gradient_checkpointing and not use_cache:
                x, aux, _ = torch.utils.checkpoint.checkpoint(
                    block, x, prev_experts, use_reentrant=False
                )
            else:
                if use_cache:
                    x, aux, new_kv = block(x, prev_experts, past_kv=past_kvs[i])
                    new_past_kvs.append(new_kv)
                else:
                    x, aux, _ = block(x, prev_experts)
            total_aux_loss += aux
            if self.cfg.use_rex and self.cfg.use_moe and hasattr(block, "moe"):
                prev_experts = block.moe.get_expert_modules()

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

        if use_cache:
            return logits, mtp_logits[0], mtp_logits[1], total_aux_loss, new_past_kvs
        return logits, mtp_logits[0], mtp_logits[1], total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
    ):
        self.eval()
        max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.cfg.generate_max_new_tokens
        )
        temperature = temperature if temperature is not None else self.cfg.generate_temperature
        top_k = top_k if top_k is not None else self.cfg.generate_top_k
        past_kvs = None
        for i in range(max_new_tokens):
            if past_kvs is not None:
                inp = input_ids[:, -1:]
            else:
                inp = input_ids[:, -self.cfg.max_ctx_len :]
            out = self.forward(inp, use_cache=True, past_kvs=past_kvs)
            logits = out[0]
            past_kvs = out[4]
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
