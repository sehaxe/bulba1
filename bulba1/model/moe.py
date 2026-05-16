"""
Mixture of Experts (MoE) module for Bulba1.

Implements token-choice and expert-choice routing with support for
shared experts and ReX (reuse) optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bulba1.model.bit_linear import (
    BitLinear,
    activation_quant_ste,
    activation_quant_ste_absmean,
    ste_b158,
)


class Expert(nn.Module):
    """Single expert layer with configurable activation function."""
    def __init__(self, d_model, hidden_dim, use_bitlinear=True, activation_bits=8,
                 activation_fn="gelu", use_absmean_down=False):
        super().__init__()
        if use_bitlinear:
            self.w1 = BitLinear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
            self.w2 = BitLinear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
            self.w3 = BitLinear(hidden_dim, d_model, bias=False, activation_bits=activation_bits)
        else:
            self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.activation_fn = activation_fn
        self.use_absmean_down = use_absmean_down
        self.use_bitlinear = use_bitlinear
        self.activation_bits = activation_bits

    def forward(self, x):
        if self.activation_fn == "relu2":
            h = F.relu(self.w1(x)).pow(2) * self.w2(x)
        elif self.activation_fn == "silu":
            h = F.silu(self.w1(x)) * self.w2(x)
        else:
            h = F.gelu(self.w1(x)) * self.w2(x)
        if self.use_bitlinear and self.use_absmean_down:
            h = activation_quant_ste_absmean(h, self.activation_bits)
        return self.w3(h)


class MoELayer(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.d_model = cfg.d_model
        self.expert_hidden = cfg.expert_hidden
        self.use_bitlinear = cfg.use_bitlinear
        self.layer_idx = layer_idx
        self.use_rex = cfg.use_rex
        self.use_grouped_gemm = getattr(cfg, "use_grouped_gemm", True)
        self.num_shared_experts = getattr(cfg, "num_shared_experts", 2)
        self.router_z_loss_coef = getattr(cfg, "router_z_loss_coef", 0.001)
        self.router_entropy_coef = getattr(cfg, "router_entropy_coef", 0.001)
        self.use_expert_choice = getattr(cfg, "use_expert_choice", False)
        self.expert_choice_capacity = getattr(cfg, "expert_choice_capacity", 0)

        abits = getattr(cfg, "bitnet_activation_bits", 8)
        activation_fn = "relu2" if getattr(cfg, "a48_use_relu2_glu", False) else "gelu"
        use_absmean_down = getattr(cfg, "use_bitnet_a48", False)

        self.shared_experts = nn.ModuleList([
            Expert(cfg.d_model, cfg.expert_hidden, cfg.use_bitlinear,
                   abits, "silu", use_absmean_down)
            for _ in range(self.num_shared_experts)
        ])

        if self.use_grouped_gemm:
            self.grouped_experts = GroupedExperts(
                cfg.num_experts, cfg.d_model, cfg.expert_hidden,
                cfg.use_bitlinear, abits, activation_fn, use_absmean_down
            )
        else:
            self.experts = nn.ModuleList([
                Expert(cfg.d_model, cfg.expert_hidden, cfg.use_bitlinear,
                       abits, activation_fn, use_absmean_down)
                for _ in range(cfg.num_experts)
            ])

        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        if self.use_rex:
            self.reuse_weight = nn.Parameter(
                torch.ones(1) * getattr(cfg, "rex_reuse_weight", 0.3)
            )

    def _forward_grouped_single_pass(self, x_rep, expert_indices, weights):
        """Один вызов grouped_experts с уже повторёнными x и индексами."""
        out = self.grouped_experts(x_rep, expert_indices)
        return out * weights.unsqueeze(-1)

    def _forward_loop_single_pass(self, x_rep, expert_indices, weights):
        out = torch.zeros(x_rep.shape[0], x_rep.shape[1], device=x_rep.device, dtype=x_rep.dtype)
        for eid in range(self.num_experts):
            mask = expert_indices == eid
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            e_out = self.experts[eid](x_rep[idx])
            w = weights[idx].unsqueeze(-1).to(out.dtype)
            out.index_add_(0, idx, e_out.to(out.dtype) * w)
        return out

    def _get_expert_output(self, x_flat, topk_idx, topk_vals):
        """
        Вычисляет выход экспертов суммированием по k, не дублируя x_flat.
        Память оптимальна, скорость почти как у единого вызова.
        """
        output = torch.zeros(x_flat.shape[0], x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]   # (B*T,)
            weights = topk_vals[:, k]     # (B*T,)

            if self.use_grouped_gemm:
                unique_e, counts = torch.unique(expert_ids, return_counts=True)
                avg_load = counts.float().mean() if len(unique_e) > 0 else 0.0
                if self.training and avg_load < 4.0:
                    out_k = self._forward_loop_single_pass(x_flat, expert_ids, weights)
                else:
                    out_k = self._forward_grouped_single_pass(x_flat, expert_ids, weights)
            else:
                out_k = self._forward_loop_single_pass(x_flat, expert_ids, weights)

            output = output + out_k

        return output

    def forward(self, x: torch.Tensor, prev_experts=None):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        logits = self.gate(x_flat)
        router_prob = F.softmax(logits, dim=-1)

        if self.use_expert_choice:
            # Expert Choice Routing (опционально, сейчас не используется)
            capacity = self.expert_choice_capacity
            if capacity <= 0:
                capacity = int(B * T * self.top_k / self.num_experts) + 1
            topk_vals, topk_idx = torch.topk(router_prob.t(), capacity, dim=1)  # (E, capacity)
            output = torch.zeros(x_flat.shape[0], x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
            for e in range(self.num_experts):
                tokens = topk_idx[e]
                if tokens.numel() == 0:
                    continue
                vals = topk_vals[e]
                expert_in = x_flat[tokens]
                if self.use_grouped_gemm:
                    e_out = self.grouped_experts(expert_in, torch.full_like(tokens, e))
                else:
                    e_out = self.experts[e](expert_in)
                output.index_add_(0, tokens, (e_out * vals.unsqueeze(-1)).to(output.dtype))
            for se in self.shared_experts:
                output = output + se(x_flat)
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = self.router_z_loss_coef * (log_z ** 2).mean()
            total_aux = z_loss
        else:
            # Token Choice
            topk_vals, topk_idx = torch.topk(router_prob, self.top_k, dim=-1)
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

            # Один проход через экспертов
            output = self._get_expert_output(x_flat, topk_idx, topk_vals)

            # Shared experts
            for se in self.shared_experts:
                output = output + se(x_flat)

            # ReX
            if self.use_rex and prev_experts is not None and len(prev_experts) > 0:
                with torch.no_grad():
                    prev_out = torch.zeros(x_flat.shape[0], x_flat.shape[1], device=x_flat.device, dtype=x_flat.dtype)
                    for k in range(self.top_k):
                        k_idx = topk_idx[:, k]   # (B*T,)
                        k_vals = topk_vals[:, k]
                        for eid in range(min(self.num_experts, len(prev_experts))):
                            mask = k_idx == eid
                            if not mask.any():
                                continue
                            idx = mask.nonzero(as_tuple=True)[0]
                            p_out = prev_experts[eid](x_flat[idx])
                            prev_out.index_add_(0, idx, (p_out * k_vals[idx].unsqueeze(-1)).to(prev_out.dtype))
                rw = torch.sigmoid(self.reuse_weight)
                output = output + prev_out * rw

            # Aux losses
            aux_loss = self.num_experts * (router_prob.mean(dim=0) ** 2).sum()
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = self.router_z_loss_coef * (log_z ** 2).mean()
            entropy = -(router_prob * torch.log(router_prob + 1e-10)).sum(dim=-1).mean()
            entropy_loss = -self.router_entropy_coef * entropy
            total_aux = aux_loss + z_loss + entropy_loss

        return output.view(B, T, D), total_aux

    def get_expert_modules(self):
        if self.use_grouped_gemm:
            class ExpertWrapper:
                def __init__(self, grouped, eid):
                    self.grouped = grouped
                    self.eid = eid
                def __call__(self, x):
                    ids = torch.full((x.size(0),), self.eid, dtype=torch.long, device=x.device)
                    return self.grouped(x, ids)
            return [ExpertWrapper(self.grouped_experts, eid) for eid in range(self.num_experts)]
        return list(self.experts)


