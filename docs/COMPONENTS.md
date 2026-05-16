# Components Reference Guide

This guide covers all model and training components in Bulba 1 with quick reference for developers.

---

## Model Components

### Core Model

| File | Class | Purpose |
|------|-------|---------|
| `model/minichat.py` | `MiniChat` | Main model container |
| `model/minichat.py` | `TiedHead` | Tied LM head (shares embedding weight) |
| `model/block.py` | `Block` | Single layer block |

### New Features (v2)

| File | Class | Purpose |
|------|-------|---------|
| `model/token_merging.py` | `TokenMerger` | Inference-time token merging |
| `model/mod.py` | `MoDGate` | Mixture of Depths dynamic gating |

### Attention

| File | Class | Purpose |
|------|-------|---------|
| `model/kda.py` | `KimiDeltaAttention` | KDA with parallel scan |
| `model/diff_attn.py` | `DiffAttention` | Differential attention with MLA |
| `model/diff_attn.py` | `RMSNorm` | RMS normalization |
| `model/diff_attn.py` | `RoPE` | Rotary position embeddings |

### State Space Model

| File | Class | Purpose |
|------|-------|---------|
| `model/mamba.py` | `MambaBlock` | Mamba-3 SSM block |

### Mixture of Experts

| File | Class | Purpose |
|------|-------|---------|
| `model/moe.py` | `MoELayer` | MoE with routing, shared experts, ReX |
| `model/moe.py` | `Expert` | Individual expert |
| `model/moe.py` | `GroupedExperts` | Batched expert computation |
| `model/moe.py` | `SharedExpert` | Always-active shared expert |

### Latent Clustering

| File | Class | Purpose |
|------|-------|---------|
| `model/mhc.py` | `MHC` | Multi-Head Latent Clustering |

### Quantization

| File | Class | Purpose |
|------|-------|---------|
| `model/bit_linear.py` | `BitLinear` | Ternary weight quantization |
| `model/bit_linear.py` | `ste_b158` | Straight-through estimator |

---

## Training Components

### Core Training

| File | Class | Purpose |
|------|-------|---------|
| `training/engine.py` | `TrainingEngine` | Main training loop (+ AutoPilot) |
| `training/optimizer.py` | `CombinedOptimizer` | Muon + AdamW combined |
| `training/optimizer.py` | `MuonOptimizer` | Muon optimizer (orthogonal init) |
| `training/ema.py` | `EMA` | Exponential moving average |
| `training/checkpoint.py` | `CheckpointManager` | Save/load with rotation |

### AutoPilot (Autonomous Training)

| File | Class | Purpose |
|------|-------|---------|
| `autonomy.py` | `AutoPilot` | Autonomous HP tuning |

**Features:**
- Modes: CALIBRATE → EXPLORE → EXPLOIT → PLATEAU → SGDR
- Dynamic LR computation
- Plateau detection
- Warm restarts

### Other Training

| File | Purpose |
|------|---------|
| `training/stages.py` | Multi-stage training |
| `training/eval.py` | Evaluation |
| `training/monitor.py` | GPU/memory monitoring |
| `training/chunked_ce.py` | Memory-efficient cross entropy |

---

## Configuration & Utilities

| File | Class | Purpose |
|------|-------|---------|
| `config.py` | `ModelConfig` | Model config dataclass |
| `tokenizer.py` | `SmartTokenizer` | Auto-detecting tokenizer trainer |
| `tokenizer.py` | `FastTokenizer` | Fast inference tokenizer |
| `tokenizer.py` | `HFTokenizer` | HuggingFace tokenizer wrapper |
| `tokenizer.py` | `TextDataset` | Iterable text dataset |
| `tokenizer.py` | `BinaryDataset` | Iterable binary dataset |
| `cli.py` | - | CLI with --auto flag |
| `orchestrator.py` | - | End-to-end pipeline |

---

## Quick Decision Guide

### Enable/Disable New Features

| Feature | Config Option | When to Enable |
|---------|---------------|----------------|
| Token Merging | `inference_merge_ratio > 0` | Long context inference |
| Mixture of Depths | `use_mixture_of_depths: true` | Dynamic depth allocation |
| Tied Embeddings | `tied_embeddings: true` | Memory efficiency (default) |
| Recurrent Blocks | `recurrent_repeats > 1` | Deeper network with fewer params |
| AutoPilot | `--auto` flag | Autonomous training |

### Memory vs Speed Trade-offs

```
Highest VRAM     ←────────────────────→     Fastest Training
   │                                            │
   ▼                                            ▼
use_mod ON                           Gradient Checkpointing OFF
Gradient Checkpointing ON              Batch size larger
Batch size smaller                      Sequence longer
Sequence shorter                        use_mamba OFF
use_bitlinear ON                        Tied Embeddings OFF
Tied Embeddings ON
```

---

## New Config Options

```yaml
# Recurrent Architecture
num_unique_blocks: 10    # Unique blocks before repeating
recurrent_repeats: 1     # How many times to repeat blocks

# Token Merging (Inference)
inference_merge_ratio: 0.3  # Ratio of tokens to merge
merge_every_n_layers: 2     # Merge frequency

# Mixture of Depths
use_mixture_of_depths: false
mod_capacity: 0.75

# Tied Embeddings (Default ON)
tied_embeddings: true
```

---

## Testing Components

```bash
# Test model
pytest tests/test_model.py -v

# Quick debug
python -c "
import torch
from bulba1 import MiniChat, ModelConfig

cfg = ModelConfig(d_model=512, n_layers=10, vocab_size=26000)
model = MiniChat(cfg)
x = torch.randint(0, 26000, (2, 32))
logits, mtp1, mtp2, aux = model(x)
print(f'Output shape: {logits.shape}')
"
```