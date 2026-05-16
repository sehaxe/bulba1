# Bulba 1 Architecture

## Overview

Bulba 1 — autonomous LLM training platform for consumer GPUs. Hybrid architecture combining Mamba-3, Kimi Delta Attention (KDA), Mixture of Experts (MoE), and BitNet quantization. Now with **AutoPilot** for autonomous hyperparameter tuning.

## Technologies Used

| Technology | Purpose | Status |
|------------|---------|--------|
| **Mamba-3** | State Space Model for long-range modeling | ✅ Active |
| **KDA (Kimi Delta Attention)** | Delta attention with parallel scan | ✅ Active |
| **MoE (Mixture of Experts)** | Efficient computation with 4 experts | ✅ Active |
| **ReX** | Reuse previous layer experts | ✅ Active |
| **MHC** | Multi-Head Latent Clustering | ✅ Active |
| **AutoPilot** | Autonomous HP tuning | ✅ Active |
| **MoD** | Mixture of Depths (dynamic depth) | ✅ Active |
| **Token Merging** | Inference-time efficiency | ✅ Active |
| **Tied Embeddings** | Share LM head with embedding | ✅ Active |
| **BitNet** | Ternary weight quantization | ✅ Active |
| **MLA** | Multi-Latent Attention (compressed KV) | ✅ Active |
| **MTP** | Multi-Token Prediction | ✅ Active |

## Model Configuration

```
┌─────────────────────────────────────────────────────────────┐
│ Bulba 1 Default Config (v2 with Recurrent Blocks)          │
├─────────────────────────────────────────────────────────────┤
│ d_model = 512           # Embedding size                  │
│ n_layers (base) = 10   # Config value                    │
│ num_unique_blocks = 4   # Unique blocks before repeating   │
│ recurrent_repeats = 3   # Times to repeat blocks           │
│ Effective layers = 12   # 4 × 3 = 12 total layers          │
│ n_heads = 8             # Attention heads                  │
│ vocab_size = 26000      # Vocabulary                        │
│ num_experts = 4         # MoE experts (3 shared + 1 routed) │
│ Total params = 26.7M    # With MoD, Token Merging, TiedEmb │
└─────────────────────────────────────────────────────────────┘
```

## New Features (v2 from Dump)

### Recurrent Blocks
- `num_unique_blocks` - unique blocks before repeating
- `recurrent_repeats` - how many times to repeat blocks
- `merge_every_n_layers` - token merging frequency

### Mixture of Depths (MoD)
- Dynamic depth allocation via MoDGate
- `use_mixture_of_depths` - enable/disable
- `mod_capacity` - capacity ratio (default 0.75)

### Token Merging
- Inference-time token merging for efficiency
- `inference_merge_ratio` - ratio of tokens to merge
- `merge_every_n_layers` - merge frequency

### Tied Embeddings
- LM head tied to embedding weight (memory efficient)
- `tied_embeddings` - enable/disable (default True)

## Layer Arrangement

Attentional pattern based on `attn_every_n_layers` (default: 4):

| Layer Type | Layers | Components |
|------------|--------|------------|
| Attention Block | 0, 4, 8 | KDA/DiffAttn → MoE → MHC |
| Mamba Block | 1-3, 5-7, 9-11 | Mamba-3 → MHC |

## Core Components

### 1. Embedding + CLR Tokens

```python
self.embedding = nn.Embedding(vocab_size, d_model)
# Optional: learnable CLR tokens
clr_tokens = nn.Parameter(torch.randn(1, num_clr_tokens, d_model))
```

### 2. Attention (KDA / DiffAttn)

- Parallel scan for efficiency (`kda_use_parallel_scan=True`)
- Double gating (`kda_double_gate=True`)
- RoPE positional embeddings
- MLA (Multi-Latent Attention) for compressed KV
- Quantized KV cache (3-bit)

### 3. Mamba-3 (State Space Model)

```python
from mamba_ssm import Mamba as Mamba3
# Linear complexity, no KV cache
```

### 4. MoE + ReX

- Shared experts (always active) + routed experts
- Top-k routing (default top-2)
- ReX: reuse previous layer experts

### 5. MHC (Multi-Head Latent Clustering)

- DeepSeek-style latent clustering
- Residual stream mixing
- `mhc_n=4`, `mhc_iterations=4`

### 6. MTP (Multi-Token Prediction)

Predict multiple tokens ahead:

```python
for i in range(num_mtp_heads):
    mtp_logits[i] = mtp_head(silu(mtp_proj(h_mtp)))
    h_mtp = silu(mtp_proj(h_mtp))
```

### 7. BitNet Quantization

- BitLinear: ternary weights ({-1, 0, +1})
- 8-bit activations
- 3-bit KV cache

## Training Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    Training Loop                           │
├────────────────────────────────────────────────────────────┤
│ 1. Data Loading                                           │
│    ├─ Infinite loader → batches                           │
│    ├─ Tokenizer → input_ids, labels                      │
│    └─ Curriculum: dynamic sequence length                │
│                                                            │
│ 2. Forward Pass (with Gradient Checkpointing)             │
│    ├─ Embedding + CLR tokens                              │
│    ├─ Blocks (with Token Merging, MoD if enabled)         │
│    ├─ RMSNorm + LM Head (tied or untied)                  │
│    └─ MTP heads                                           │
│                                                            │
│ 3. Loss Computation                                        │
│    ├─ CrossEntropy (main)                                 │
│    ├─ MTP losses (t+1, t+2)                              │
│    └─ Router auxiliary losses                             │
│                                                            │
│ 4. Backward Pass (BF16 AMP)                               │
│                                                            │
│ 5. Optimizer Step                                         │
│    ├─ Muon (Newton-Schulz) for large layers               │
│    └─ AdamW for norms/embeddings                          │
│                                                            │
│ 6. AutoPilot (if --auto enabled)                         │
│    ├─ compute_lr(step) - dynamic LR                      │
│    └─ step(step, loss) - adjust hyperparameters           │
│                                                            │
│ 7. Checkpointing                                          │
└────────────────────────────────────────────────────────────┘
```

## AutoPilot (Autonomous Training)

Enable with `--auto` flag:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

AutoPilot features:
- **Dynamic LR**: Adjusts learning rate based on training progress
- **Plateau Detection**: Detects when loss plateaus
- **Warm Restarts**: Restarts with new LR when stuck
- **Hyperparameter Tuning**: Adjusts weight decay, gradient noise

```python
# AutoPilot modes: CALIBRATE → EXPLORE → EXPLOIT → PLATEAU → SGDR
engine = TrainingEngine(model, cfg, tokenizer, auto_mode=True)
# Uses autopilot.compute_lr(step) instead of cosine schedule
# Calls autopilot.step(step, loss) to adjust params
```

## VRAM Optimization

| Technique | VRAM Saved | Trade-off |
|-----------|------------|-----------|
| BF16 AMP | ~50% | None |
| Gradient Checkpointing | ~30% | 10-20% slower |
| MLA (latent KV) | ~40% | Slight quality loss |
| Token Merging (inference) | ~30% | Longer inference |
| Tied Embeddings | ~10% | Shares weights |
| use_mamba=False | ~1 GB | Lose SSM benefits |
| use_bitlinear=True | ~20% | Quantization noise |

## Default VRAM Usage (RTX 5060 Ti 16GB)

With new smaller config (d_model=512, n_layers=10):

| batch | seq | VRAM |
|-------|-----|------|
| 32 | 512 | ~12 GB |
| 24 | 512 | ~10 GB |
| 16 | 512 | ~7 GB |

## References

- Mamba-3: https://arxiv.org/abs/2603.15569
- BitNet: https://arxiv.org/abs/2309.05512
- DeepSeek-MoE: https://arxiv.org/abs/2401.06066
- RoPE: https://arxiv.org/abs/2104.09864
- Kimi k1.5: https://arxiv.org/abs/2501.12598
- BitNet b1.58: https://arxiv.org/abs/2402.17762