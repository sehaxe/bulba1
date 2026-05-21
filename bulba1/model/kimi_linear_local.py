"""
Local Attention module for Kimi Linear (sliding window for exact recall).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    """Sliding window attention for exact recall (critical for coding)."""
    def __init__(self, cfg, window_size: int = 512):
        super().__init__()
        self.window_size = window_size
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: (B, H, T, D)
            mask: optional attention mask
        
        Returns:
            output: (B, H, T, D)
            z_loss: scalar (for stability)
        """
        B, H, T, D = q.shape
        
        # Create sliding window mask
        window_mask = torch.ones(T, T, dtype=torch.bool, device=q.device)
        for i in range(T):
            start = max(0, i - self.window_size + 1)
            window_mask[i, start:i+1] = False
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        if mask is not None:
            window_mask = window_mask | mask
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(window_mask, float('-inf'))
        
        # z_loss for stability
        z_loss = (torch.logsumexp(scores, dim=-1) ** 2).mean()
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        return out, z_loss
