# Developer Guide

## Project Structure

```
bulba1-python/
├── bulba1/                    # Main package
│   ├── model/                 # Model architecture
│   │   ├── minichat.py        # Main model (MiniChat, TiedHead)
│   │   ├── block.py           # Layer block (Block)
│   │   ├── kda.py             # Kimi Delta Attention
│   │   ├── diff_attn.py       # Differential Attention
│   │   ├── mamba.py           # Mamba-3 SSM
│   │   ├── moe.py             # Mixture of Experts + ReX
│   │   ├── mhc.py             # Multi-Head Latent Clustering
│   │   ├── bit_linear.py      # BitNet quantization
│   │   ├── token_merging.py   # Token merging (new)
│   │   └── mod.py             # MoDGate (new)
│   │
│   ├── training/               # Training pipeline
│   │   ├── engine.py          # Main training loop + AutoPilot
│   │   ├── optimizer.py       # Muon + AdamW
│   │   ├── checkpoint.py      # Checkpoint management
│   │   ├── ema.py             # Exponential moving average
│   │   ├── stages.py          # Multi-stage training
│   │   ├── eval.py            # Evaluation
│   │   ├── monitor.py         # System monitoring
│   │   └── chunked_ce.py      # Memory-efficient CE
│   │
│   ├── autonomy.py            # AutoPilot (new v2 feature)
│   ├── config.py             # ModelConfig
│   ├── tokenizer.py          # Custom tokenizer
│   ├── orchestrator.py       # End-to-end pipeline
│   └── cli.py                # CLI interface (with --auto)
│
├── configs/
│   └── default.yaml           # Main config (smaller model: 512d, 10 layers)
│
├── tools/                     # Profiling & visualization
├── scripts/                   # Data scripts
├── docs/                     # Documentation
└── tests/                    # Test suite
```

## Development Setup

```bash
# Install dependencies
uv sync --extra cuda --extra dev

# Verify
python -c "from bulba1 import MiniChat; print('OK')"

# Run CLI
python -m bulba1.cli --help
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=bulba1 --cov-report=html
```

## New Features (v2 from Dump)

### 1. AutoPilot (Autonomous Training)

Enable with `--auto` flag:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

In code:

```python
from bulba1.training.engine import TrainingEngine

# With AutoPilot
engine = TrainingEngine(model, cfg, tokenizer, auto_mode=True)

# Without AutoPilot (default)
engine = TrainingEngine(model, cfg, tokenizer, auto_mode=False)
```

AutoPilot adjusts:
- Learning rate dynamically
- Detects plateaus
- Triggers warm restarts

### 2. Token Merging

Enable in config:

```yaml
model:
  inference_merge_ratio: 0.3  # 30% token merging at inference
  merge_every_n_layers: 2     # Merge every 2 layers
```

### 3. Mixture of Depths (MoD)

Enable in config:

```yaml
model:
  use_mixture_of_depths: true
  mod_capacity: 0.75  # Keep 75% of tokens
```

### 4. Recurrent Blocks

```yaml
model:
  num_unique_blocks: 10  # Unique blocks
  recurrent_repeats: 2  # Repeat twice = 20 effective layers
```

### 5. Tied Embeddings

```yaml
model:
  tied_embeddings: true  # Default, saves memory
```

## Adding New Features

### 1. New Model Component

```python
# model/my_module.py
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.linear(x)
```

### 2. Enable AutoPilot

The AutoPilot is already integrated in engine.py. Just use `--auto` flag:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

## Debugging

### Print Model Structure

```python
# List all layers
for name, module in model.named_modules():
    print(name)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total/1e6:.1f}M")
```

### Debug VRAM

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Debug AutoPilot

```python
# Check if AutoPilot is active
print(f"Auto mode: {engine.auto_mode}")
print(f"Autopilot: {engine.autopilot}")

# Get current state
if engine.autopilot:
    print(f"Mode: {engine.autopilot.state.mode}")
    print(f"LR: {engine.autopilot.current_lr}")
```

## Common Issues

### OOM

```python
# Solution 1: Reduce model size
d_model: 512  # from 768
n_layers: 10  # from 16

# Solution 2: Reduce batch
batch_size: 16

# Solution 3: Enable gradient checkpointing
use_gradient_checkpointing: true

# Solution 4: Disable features
use_mamba: false
use_bitlinear: false
use_moe: false
```

### Training Instability

```python
# Use AutoPilot for automatic adjustment
python -m bulba1.cli --config configs/default.yaml --auto
```

### Checkpoint Issues

```python
# Verify checkpoint
import torch
ckpt = torch.load("checkpoints/.../model.pt", map_location="cpu")
print(f"Step: {ckpt.get('step', 'N/A')}")

# Resume with AutoPilot state
python -m bulba1.cli --resume --checkpoint-dir checkpoints/... --auto
```

## Git Workflow

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes
# ...

# Commit
git add .
git commit -m "Add my feature"

# Push
git push origin feature/my-feature
```

## Code Review Checklist

- [ ] Type hints on public functions
- [ ] Docstrings on new features
- [ ] Tests pass
- [ ] VRAM usage acceptable (<14GB)
- [ ] Works with CUDA
- [ ] Config options have defaults
- [ ] Works with and without --auto

## Performance Tuning

| Optimization | Command/Config | Impact |
|--------------|----------------|--------|
| torch.compile | `--compile` | ~20% faster |
| Gradient checkpointing | `use_gradient_checkpointing: true` | -30% VRAM |
| Smaller model | `d_model: 512, n_layers: 10` | -40% VRAM |
| Tied embeddings | `tied_embeddings: true` | -10% VRAM |
| Token merging | `inference_merge_ratio: 0.3` | -30% inference VRAM |
| Disable MoE | `use_moe: false` | -2 GB VRAM |

## Resources

- PyTorch docs: https://pytorch.org/docs/
- Mamba-3: https://github.com/state-spaces/mamba
- BitNet: https://arxiv.org/abs/2309.05512