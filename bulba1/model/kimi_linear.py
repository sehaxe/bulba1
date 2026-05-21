"""
Kimi Linear Attention - Chunkwise Parallel Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.bit_linear import make_linear, quantize_ste_absmax
from bulba1.model.diff_attn import RMSNorm, RoPE

try:
    from fla.ops.kda import chunk_kda
    FLA_AVAILABLE = True
    print("✅ FLA kernel available (Triton-accelerated)")
except ImportError:
    FLA_AVAILABLE = False
    print("⚠️  FLA kernel not available, using PyTorch chunkwise fallback")

class ChunkwiseKDA_PyTorch(nn.Module):
    def __init__(self, chunk_size: int = 64):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, q, k, v, g, beta, initial_state=None):
        B, H, T, D_k = q.shape
        D_v = v.shape[-1]
        C = self.chunk_size

        if T % C != 0:
            pad_len = C - (T % C)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            g = F.pad(g, (0, 0, 0, pad_len))
            beta = F.pad(beta, (0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T

        N = T_padded // C
        q = q.view(B, H, N, C, D_k)
        k = k.view(B, H, N, C, D_k)
        v = v.view(B, H, N, C, D_v)
        g = g.view(B, H, N, C, D_k)
        beta = beta.view(B, H, N, C)

        g_cumsum = g.cumsum(dim=-2)
        if initial_state is None:
            state = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=q.dtype)
        else:
            state = initial_state.clone()

        mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        
        outputs = []
        for chunk_idx in range(N):
            q_c = q.unbind(2)[chunk_idx]
            k_c = k.unbind(2)[chunk_idx]
            v_c = v.unbind(2)[chunk_idx]
            g_c = g.unbind(2)[chunk_idx]
            g_cum_c = g_cumsum.unbind(2)[chunk_idx]
            beta_c = beta.unbind(2)[chunk_idx]

            q_scaled = q_c * g_cum_c.exp()
            A = torch.einsum('bhic,bhjc->bhij', q_scaled, k_c)
            A = A.masked_fill(mask, 0.0)

            # FIX: Mask upper triangle with -inf BEFORE exp to prevent 0 * inf = NaN
            g_diff = g_cum_c.unsqueeze(-2) - g_cum_c.unsqueeze(-3)
            g_diff = g_diff.masked_fill(mask_expanded, float('-inf'))
            decay = g_diff.exp().prod(dim=-1)
            decay = decay.nan_to_num(0.0)
            
            A = A * decay
            o_intra = torch.einsum('bhij,bhjd->bhid', A, v_c)
            o_inter = torch.einsum('bhic,bhcd->bhid', q_scaled, state)
            o_chunk = o_intra + o_inter
            outputs.append(o_chunk)

            decay_final = g_cum_c[:, :, -1, :].exp()
            k_weighted = k_c * beta_c.unsqueeze(-1)
            state = state * decay_final.unsqueeze(-1)
            state = state + torch.einsum('bhci,bhcd->bhid', k_weighted, v_c)

        output = torch.cat(outputs, dim=2)
        if T_padded != T:
            output = output[:, :, :T, :]
        return output, state

class KimiLinear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.local_window = getattr(cfg, "kimi_local_window", 512)
        self.chunk_size = getattr(cfg, "kimi_chunk_size", 64)

        self.q_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = make_linear(cfg, cfg.d_model, cfg.d_model, bias=False)

        self.gate_dim = getattr(cfg, "kimi_gate_dim", 16)
        self.use_double_gate = getattr(cfg, "kimi_double_gate", True)
        if self.use_double_gate:
            self.gate_proj = make_linear(cfg, cfg.d_model, self.n_heads * self.gate_dim * 2, bias=False)
            self.gate_out = nn.Linear(self.gate_dim * 2, 2, bias=False)
        else:
            self.gate_proj = make_linear(cfg, cfg.d_model, self.n_heads * self.gate_dim, bias=False)
            self.gate_out = nn.Linear(self.gate_dim, 1, bias=False)

        self.beta_proj = make_linear(cfg, cfg.d_model, self.n_heads, bias=False)

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        self.use_rope = getattr(cfg, "kimi_use_rope", True)
        if self.use_rope:
            self.rope = RoPE(
                self.head_dim,
                getattr(cfg, "max_ctx_len", 32768),
                getattr(cfg, "rope_theta", 10000.0)
            )

        from bulba1.model.kimi_linear_local import LocalAttention
        self.local_attn = LocalAttention(cfg, window_size=self.local_window)

        self.use_quantized_kv_cache = getattr(cfg, "use_quantized_kv_cache", True)
        self.kv_cache_bits = getattr(cfg, "kv_cache_bits", 3)
        self.mix_weight = nn.Parameter(torch.tensor(0.7))
        self.chunkwise_kda = ChunkwiseKDA_PyTorch(chunk_size=self.chunk_size)

    def _compute_gates_and_beta(self, x):
        B, T, _ = x.shape
        gate_logits = self.gate_proj(x).view(B, T, self.n_heads, -1)
        if self.use_double_gate:
            gate = torch.sigmoid(self.gate_out(gate_logits))
            gate_f = gate[..., 0]
        else:
            gate_f = torch.sigmoid(self.gate_out(gate_logits).squeeze(-1))
        g = torch.log(gate_f + 1e-6).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        return g, beta

    def _forward_fla(self, q, k, v, g, beta):
        output, _ = chunk_kda(
            q=q.transpose(1, 2), k=k.transpose(1, 2), v=v.transpose(1, 2),
            g=g.transpose(1, 2), beta=beta.transpose(1, 2),
            initial_state=None, chunk_size=self.chunk_size
        )
        return output.transpose(1, 2)

    def _forward_chunkwise(self, q, k, v, g, beta):
        output, _ = self.chunkwise_kda(q, k, v, g, beta, initial_state=None)
        return output

    def forward(self, x, mask=None, past_kv=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        q, k = self.norm_q(q), self.norm_k(k)
        if self.use_quantized_kv_cache:
            k = quantize_ste_absmax(k, self.kv_cache_bits)
            v = quantize_ste_absmax(v, self.kv_cache_bits)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.use_rope:
            q, k = self.rope(q, T), self.rope(k, T)

        g, beta = self._compute_gates_and_beta(x)

        if FLA_AVAILABLE:
            try:
                linear_out = self._forward_fla(q, k, v, g, beta)
            except Exception:
                linear_out = self._forward_chunkwise(q, k, v, g, beta)
        else:
            linear_out = self._forward_chunkwise(q, k, v, g, beta)

        local_out, z_loss = self.local_attn(q, k, v, mask)
        alpha = torch.sigmoid(self.mix_weight)
        out = alpha * linear_out + (1 - alpha) * local_out

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)
        new_kv = (k, v) if past_kv is None else None
        return out, new_kv, z_loss