"""
Kimi Delta Attention - Chunkwise Parallel Implementation
Based on arXiv:2510.26692 (Moonshot AI).
Paper: https://arxiv.org/abs/2510.26692
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.bit_linear import make_linear, quantize_ste_absmax
from bulba1.model.diff_attn import RMSNorm, RoPE

class ChunkwiseKDA(nn.Module):
    """
    Chunkwise parallel KDA kernel (PyTorch fallback).
    """
    def __init__(self, chunk_size: int = 64):
        super().__init__()
        self.C = chunk_size

    def forward(self, q, k, v, g, beta, initial_state=None):
        B, H, T, Dk = q.shape
        Dv = v.shape[-1]
        C = self.C

        pad = (C - T % C) % C
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        T_pad = T + pad
        N = T_pad // C

        q = q.view(B, H, N, C, Dk)
        k = k.view(B, H, N, C, Dk)
        v = v.view(B, H, N, C, Dv)
        g = g.view(B, H, N, C, Dk)
        beta = beta.view(B, H, N, C)

        gc = g.cumsum(-2)
        S = initial_state if initial_state is not None else torch.zeros(B, H, Dk, Dv, device=q.device, dtype=q.dtype)

        q_chunks = q.unbind(2)
        k_chunks = k.unbind(2)
        v_chunks = v.unbind(2)
        gc_chunks = gc.unbind(2)
        beta_chunks = beta.unbind(2)
        
        # Pre-compute causal mask
        causal = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), 1).logical_not()
        causal_mask_expanded = causal.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        outs = []
        for i in range(N):
            qi = q_chunks[i]
            ki = k_chunks[i]
            vi = v_chunks[i]
            gi = gc_chunks[i]
            bi = beta_chunks[i]

            qs = qi * gi.exp()
            A = torch.einsum("bhic,bhjc->bhij", qs, ki)
            A = A * causal.unsqueeze(0).unsqueeze(0)

            # FIX: Mask upper triangle with -inf BEFORE exp to prevent 0 * inf = NaN
            g_diff = gi.unsqueeze(-2) - gi.unsqueeze(-3)
            g_diff = g_diff.masked_fill(~causal_mask_expanded, float('-inf'))
            decay = g_diff.exp().prod(-1)
            decay = decay.nan_to_num(0.0)  # Safety fallback
            
            A = A * decay
            o_intra = torch.einsum("bhij,bhjd->bhid", A, vi)
            o_inter = torch.einsum("bhic,bhcd->bhid", qs, S)
            outs.append(o_intra + o_inter)

            decay_final = gi[:, :, -1, :].exp().unsqueeze(-1)
            S = S * decay_final + torch.einsum("bhci,bhcd->bhid", ki * bi.unsqueeze(-1), vi)

        output = torch.cat(outs, dim=2)
        if pad > 0:
            output = output[:, :, :T, :]
        return output, S

class KimiDeltaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.gate_dim = getattr(cfg, "kda_gate_dim", 16)
        self.use_double_gate = getattr(cfg, "kda_double_gate", True)

        self.q_proj = make_linear(cfg, self.d_model, self.d_model, bias=False)
        self.k_proj = make_linear(cfg, self.d_model, self.d_model, bias=False)
        self.v_proj = make_linear(cfg, self.d_model, self.d_model, bias=False)
        self.o_proj = make_linear(cfg, self.d_model, self.d_model, bias=False, quantize_input=False)

        gate_out_dim = self.gate_dim * 2 if self.use_double_gate else self.gate_dim
        self.g_proj = make_linear(cfg, self.d_model, self.n_heads * gate_out_dim, bias=False)
        self.g_out = nn.Linear(gate_out_dim, 2 if self.use_double_gate else 1, bias=False)

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        self.use_rope = getattr(cfg, "kda_use_rope", False)
        if self.use_rope:
            self.rope = RoPE(
                dim=self.head_dim,
                max_seq_len=getattr(cfg, "max_ctx_len", 4096),
                theta=getattr(cfg, "rope_theta", 10000.0),
            )

        self.use_quantized_kv = getattr(cfg, "use_quantized_kv_cache", True)
        self.kv_bits = getattr(cfg, "kv_cache_bits", 3)
        self.chunk_size = getattr(cfg, "kda_chunk_size", 64)
        self.kda_kernel = ChunkwiseKDA(chunk_size=self.chunk_size)

    def _compute_gates_and_beta(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        gate_logits = self.g_proj(x).view(B, T, self.n_heads, -1)
        if self.use_double_gate:
            gates = torch.sigmoid(self.g_out(gate_logits))
            forget_gate = gates[..., 0]
            input_gate = gates[..., 1]
            log_forget = torch.log(forget_gate + 1e-6)
            beta = input_gate * (1.0 - forget_gate)
        else:
            forget_gate = torch.sigmoid(self.g_out(gate_logits).squeeze(-1))
            log_forget = torch.log(forget_gate + 1e-6)
            beta = 1.0 - forget_gate

        g = log_forget.transpose(1, 2).unsqueeze(-1)
        g_cum = g.cumsum(dim=2)
        g_cum = g_cum.expand(-1, -1, -1, self.head_dim)
        beta = beta.transpose(1, 2)
        return g_cum, beta

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if self.use_quantized_kv and not self.training:
            k = quantize_ste_absmax(k, self.kv_bits)
            v = quantize_ste_absmax(v, self.kv_bits)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, T)
            k = self.rope(k, T)

        g_cum, beta = self._compute_gates_and_beta(x)
        output, final_state = self.kda_kernel(q, k, v, g_cum, beta, initial_state=None)

        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.o_proj(output)

        new_kv = (k, v) if past_kv is None else None
        z_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, new_kv, z_loss