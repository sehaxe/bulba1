import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.bit_linear import (
    BitLinear,
    ste_b158,
    make_linear,
    activation_quant_ste,
    activation_quant_ste_absmean,
)


class Expert(nn.Module):
    # (без изменений)
    def __init__(self, d_model, hidden_dim, use_bitlinear=True, activation_bits=8,
                 use_relu2=False, use_absmean_down=False):
        super().__init__()
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.w1 = (Linear(d_model, hidden_dim, activation_bits=activation_bits)
                   if use_bitlinear else Linear(d_model, hidden_dim, bias=False))
        self.w2 = (Linear(d_model, hidden_dim, activation_bits=activation_bits)
                   if use_bitlinear else Linear(d_model, hidden_dim, bias=False))
        self.w3 = (Linear(hidden_dim, d_model, activation_bits=activation_bits)
                   if use_bitlinear else Linear(hidden_dim, d_model, bias=False))
        self.use_relu2 = use_relu2
        self.use_absmean_down = use_absmean_down
        self.use_bitlinear = use_bitlinear
        self.activation_bits = activation_bits

    def forward(self, x):
        if self.use_relu2:
            h = F.relu(self.w1(x)).pow(2) * self.w2(x)
        else:
            h = F.silu(self.w1(x)) * self.w2(x)
        if self.use_bitlinear and self.use_absmean_down:
            h = activation_quant_ste_absmean(h, self.activation_bits)
        return self.w3(h)


class GroupedExperts(nn.Module):
    # (без изменений)
    def __init__(self, num_experts, d_model, hidden_dim, use_bitlinear=True,
                 activation_bits=8, use_relu2=False, use_absmean_down=False):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.use_bitlinear = use_bitlinear
        self.activation_bits = activation_bits
        self.use_relu2 = use_relu2
        self.use_absmean_down = use_absmean_down

        self.w1 = nn.Parameter(torch.randn(num_experts, d_model, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, d_model, hidden_dim))
        self.w3 = nn.Parameter(torch.randn(num_experts, hidden_dim, d_model))

        self.use_bias = not use_bitlinear
        if self.use_bias:
            self.b1 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
            self.b2 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
            self.b3 = nn.Parameter(torch.zeros(num_experts, d_model))

        self.reset_parameters()

    def reset_parameters(self):
        for p in [self.w1, self.w2, self.w3]:
            nn.init.kaiming_uniform_(p, a=5**0.5)

    def forward(self, x, expert_ids):
        w1 = ste_b158(self.w1) if self.use_bitlinear else self.w1
        w2 = ste_b158(self.w2) if self.use_bitlinear else self.w2
        w3 = ste_b158(self.w3) if self.use_bitlinear else self.w3

        unique_experts, counts = torch.unique(expert_ids, return_counts=True)
        num_active = unique_experts.numel()
        if num_active == 0:
            return torch.zeros_like(x)

        max_tokens = counts.max().item()
        D, H = self.d_model, self.hidden_dim

        padded = torch.zeros(num_active, max_tokens, D, device=x.device, dtype=x.dtype)
        reverse_idx = torch.empty_like(expert_ids)

        for i, eid in enumerate(unique_experts):
            mask = expert_ids == eid
            n = mask.sum()
            padded[i, :n] = x[mask]
            reverse_idx[mask.nonzero(as_tuple=True)[0]] = i * max_tokens + torch.arange(
                n, device=x.device
            )

        ew1 = w1[unique_experts]
        ew2 = w2[unique_experts]
        ew3 = w3[unique_experts]

        if self.use_bitlinear:
            padded = activation_quant_ste(padded, self.activation_bits)

        h1 = torch.bmm(padded, ew1)
        h2 = torch.bmm(padded, ew2)
        if self.use_bias:
            h1 = h1 + self.b1[unique_experts].unsqueeze(1)
            h2 = h2 + self.b2[unique_experts].unsqueeze(1)
        if self.use_relu2:
            h = F.relu(h1).pow(2) * h2
        else:
            h = F.silu(h1) * h2
        if self.use_bitlinear and self.use_absmean_down:
            h = activation_quant_ste_absmean(h, self.activation_bits)
        out = torch.bmm(h, ew3)
        if self.use_bias:
            out = out + self.b3[unique_experts].unsqueeze(1)

        flat_out = out.view(-1, D)
        return flat_out[reverse_idx]


class SharedExpert(nn.Module):
    # (без изменений)
    def __init__(self, d_model, hidden_dim, use_bitlinear=True, activation_bits=8,
                 use_relu2=False, use_absmean_down=False):
        super().__init__()
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.w1 = (Linear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
                   if use_bitlinear else Linear(d_model, hidden_dim, bias=False))
        self.w2 = (Linear(d_model, hidden_dim, bias=False, activation_bits=activation_bits)
                   if use_bitlinear else Linear(d_model, hidden_dim, bias=False))
        self.w3 = (Linear(hidden_dim, d_model, bias=False, activation_bits=activation_bits)
                   if use_bitlinear else Linear(hidden_dim, d_model, bias=False))
        self.use_relu2 = use_relu2
        self.use_absmean_down = use_absmean_down
        self.use_bitlinear = use_bitlinear
        self.activation_bits = activation_bits

    def forward(self, x):
        if self.use_relu2:
            h = F.relu(self.w1(x)).pow(2) * self.w2(x)
        else:
            h = F.silu(self.w1(x)) * self.w2(x)
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
        use_relu2 = getattr(cfg, "a48_use_relu2_glu", False)
        use_absmean_down = getattr(cfg, "use_bitnet_a48", False)

        self.shared_experts = nn.ModuleList([
            SharedExpert(cfg.d_model, cfg.expert_hidden, cfg.use_bitlinear,
                         abits, use_relu2, use_absmean_down)
            for _ in range(self.num_shared_experts)
        ])

        if self.use_grouped_gemm:
            self.grouped_experts = GroupedExperts(
                cfg.num_experts, cfg.d_model, cfg.expert_hidden,
                cfg.use_bitlinear, abits, use_relu2, use_absmean_down
            )
        else:
            self.experts = nn.ModuleList([
                Expert(cfg.d_model, cfg.expert_hidden, cfg.use_bitlinear,
                       abits, use_relu2, use_absmean_down)
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