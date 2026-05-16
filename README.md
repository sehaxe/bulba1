# Bulba 1 — Autonomous LLM Training

Self-hosted LLM training on consumer GPU. **~67M params** — runs on RTX 3060+.

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | Mamba-3 SSD + KDA + MoE |
| **AutoPilot** | Autonomous HP tuning (NEW v2) |
| **Mixture of Depths** | Dynamic depth allocation |
| **Token Merging** | Inference-time efficiency |
| **Tied Embeddings** | Memory-efficient LM head |
| **Muon + AdamW** | Custom optimizer |
| **Gradient Checkpointing** | Memory-efficient training |
| **Telegram Bot** | Monitor training, generate text |

## Quick Start

```bash
# Install
make install

# Download data
make data

# Build dataset
make build

# Train (normal)
make train

# Train with AutoPilot (autonomous HP tuning)
python -m bulba1.cli --config configs/default.yaml --auto
```

## Configuration

Default config: **512 d_model, 10 layers, 32 batch**

```yaml
model:
  d_model: 512
  n_layers: 10
  num_experts: 4

training:
  batch_size: 32
  seq_len: 512
  use_gradient_checkpointing: true
```

## AutoPilot (Autonomous Training)

Enable autonomous hyperparameter tuning:

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

AutoPilot automatically:
- Adjusts learning rate
- Detects plateaus
- Triggers warm restarts
- Optimizes weight decay, gradient noise

## VRAM Usage

| Config | VRAM (RTX 5060 Ti) |
|--------|-------------------|
| batch=32, seq=512 | ~12 GB |
| batch=24, seq=512 | ~10 GB |
| batch=16, seq=512 | ~7 GB |

## Project Structure

```
bulba1-python/
├── bulba1/
│   ├── model/              # Model architecture
│   │   ├── minichat.py     # Main model
│   │   ├── kda.py          # KDA attention
│   │   ├── moe.py          # Mixture of Experts
│   │   ├── mamba.py        # Mamba-3 SSM
│   │   └── token_merging.py  # Token merging (new)
│   ├── training/
│   │   ├── engine.py       # Training loop + AutoPilot
│   │   ├── optimizer.py    # Muon + AdamW
│   │   └── checkpoint.py   # Checkpointing
│   ├── autonomy.py         # AutoPilot (new v2)
│   └── cli.py              # CLI with --auto flag
├── configs/
│   └── default.yaml        # Main config
├── tools/
│   ├── deep_profile.py    # GPU profiling
│   └── log_viz.py         # Training visualization
└── docs/                  # Documentation
```

## Telegram Bot

- `/status` — Training status
- `/gpu` — GPU info
- `/logs` — Last log lines
- `/chat [text]` — Generate text

**Admin**: `/train`, `/stop`, `/restart`

## Requirements

- Python 3.12+
- CUDA 12.1+
- 12GB VRAM (minimum)
- 16GB RAM

## Git

https://codeberg.org/quazder/bulba1