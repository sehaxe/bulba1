# Developer Guide

## Project Structure

```
bulba1-python/
├── bulba.py                    # Entry point
├── run_225m.py            # Main training script
├── pyproject.toml             # Dependencies
│
├── bulba1/                    # Main package
│   ├── cli.py                 # CLI interface
│   ├── utils/
│   │   └── config.py          # ModelConfig dataclass
│   ├── model/
│   │   ├── minichat.py        # MiniChat model
│   │   ├── block.py           # Transformer block
│   │   ├── diff_attn.py       # Differential attention
│   │   ├── moe.py             # MoE + ReX
│   │   ├── mamba.py           # Mamba-2 SSD
│   │   ├── bit_linear.py      # BitNet quantization
│   │   ├── kda.py             # Kimi Delta Attention
│   │   └── mhc.py             # Manifold Hyper-Connections
│   ├── training/
│   │   ├── engine.py          # Training loop
│   │   ├── optimizer.py       # Muon + AdamW
│   │   ├── checkpoint.py      # Save/load
│   │   ├── autotuner.py       # Hardware auto-config
│   │   └── ema.py             # Exponential moving average
│   └── data/
│       └── tokenizer.py       # Tokenizer
│
├── telegram_bot/
│   └── bot.py                 # Telegram bot
│
├── tools/
│   ├── log_viz.py             # Plot training curves
│   ├── benchmark.py           # Performance benchmarking
│   └── auto_train.py          # Auto-training script
│
├── data/
│   ├── tokenizer_fast.json    # Trained tokenizer
│   └── train/                 # Training data (.txt files)
│
└── docs/
    ├── ARCHITECTURE.md        # Model architecture
    └── DEVELOPER_GUIDE.md    # This file
```

## Development Setup

```bash
# Clone and setup
git clone https://github.com/your-repo/bulba1-python.git
cd bulba1-python

# Install dependencies
uv sync

# Verify
python bulba.py --help
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_model.py

# With coverage
pytest --cov=bulba1
```

## Adding New Features

### 1. New Model Component

Создай файл в `bulba1/model/`:

```python
# bulba1/model/my_module.py
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ...

    def forward(self, x):
        # ...
        return output
```

Добавь в `Block` в `block.py`:

```python
from bulba1.model.my_module import MyModule

class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        # ...
        self.my_module = MyModule(cfg)
```

### 2. New Training Feature

Добавь в `TrainingEngine` в `training/engine.py`:

```python
def train_step(self, batch):
    # ... existing code
    
    # New feature
    if self.cfg.my_new_feature:
        do_something()
```

### 3. New CLI Option

Добавь в `bulba1/cli.py`:

```parser.add_argument("--my-flag", action="store_true", help="My new feature")```

И в `main()`:

```python
if args.my_flag:
    cfg.my_new_feature = True
```

## Code Style

```python
# Use type hints
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...

# Docstrings for public API
def train(self, loader):
    """Train model for one epoch.
    
    Args:
        loader: DataLoader with training batches
        
    Returns:
        Average loss for the epoch
    """
    ...

# Use dataclasses for config
@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 16
    ...
```

## Testing New Components

```python
# tests/test_my_module.py
import pytest
import torch
from bulba1.utils.config import ModelConfig
from bulba1.model.my_module import MyModule

def test_forward():
    cfg = ModelConfig(d_model=768)
    module = MyModule(cfg)
    x = torch.randn(2, 10, 768)
    out = module(x)
    assert out.shape == x.shape
```

## Debugging

```python
# Enable torch debug mode
torch.autograd.set_detect_anomaly(True)

# Print gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Profile forward pass
with torch.profiler.profile(...) as prof:
    output = model(input_ids)
```

## Common Issues

### OOM (Out of Memory)

```python
# Solution 1: Reduce batch
cfg.batch_size = 3

# Solution 2: Gradient checkpointing
cfg.use_gradient_checkpointing = True

# Solution 3: Disable features
cfg.use_mamba = False
cfg.use_bitlinear = False
```

### Slow Training

```python
# Enable torch.compile
python bulba.py --compile

# Reduce sequence length
python bulba.py --seq-len 256
```

### Checkpoint Issues

```python
# Verify checkpoint
import torch
ckpt = torch.load("checkpoints/.../model.pt", map_location="cpu")
print(ckpt.keys())

# Resume from checkpoint
python bulba.py --resume --checkpoint-dir checkpoints/...
```

## Performance Profiling

```bash
# GPU utilization
watch -n 1 nvidia-smi

# PyTorch profiler
python -c "
import torch
from bulba1 import MiniChat
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA]
) as prof:
    model(input_ids)
print(prof.key_averages().table())
"
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

# Create PR
gh pr create
```

## Code Review Checklist

- [ ] Type hints on public functions
- [ ] Docstrings on new features
- [ ] No `as any` or `@ts-ignore`
- [ ] Tests pass
- [ ] VRAM usage acceptable (<14GB)
- [ ] Works on CPU fallback

## Resources

- PyTorch docs: https://pytorch.org/docs/
- CUDA best practices: https://pytorch.org/docs/stable/notes/cuda.html
- Model training tips: https://pytorch.org/tutorials/recipes/recipes/speed_optimization.html