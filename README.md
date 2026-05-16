# Bulba 1 — Autonomous LLM Training Platform

<p align="center">
  <img src="https://img.shields.io/badge/Parameters-67M-blue" alt="Parameters">
  <img src="https://img.shields.io/badge/VRAM-12GB-green" alt="VRAM">
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License">
</p>

Self-hosted LLM training platform for consumer GPUs. Hybrid architecture combining Mamba-3 SSM, Kimi Delta Attention (KDA), Mixture of Experts (MoE), and BitNet quantization. Now with **AutoPilot** for autonomous hyperparameter tuning.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [AutoPilot](#autopilot-autonomous-training)
- [Auto SFT/DPO Pipeline](#auto-sftdpo-pipeline)
- [Training](#training)
- [VRAM Optimization](#vram-optimization)
- [Telegram Bot](#telegram-bot)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Git](#git)

---

## Features

### Core Architecture
| Feature | Description |
|---------|-------------|
| **Mamba-3 SSM** | State-space model for long-range dependencies |
| **Kimi Delta Attention (KDA)** | Linear attention with parallel scan |
| **Mixture of Experts (MoE)** | Efficient conditional computation with ReX routing |
| **DeepSeek MHC** | Multi-head cached attention mechanism |
| **DiffAttention** | Differential attention with MLA |

### Advanced Techniques
| Feature | Description |
|---------|-------------|
| **AutoPilot** | Autonomous HP tuning (modes: CALIBRATE → EXPLORE → EXPLOIT → PLATEAU → SGDR) |
| **Mixture of Depths (MoD)** | Dynamic depth allocation per token |
| **Token Merging** | Inference-time token merging for efficiency |
| **Tied Embeddings** | Share weights between embedding and LM head |
| **BitNet Quantization** | 4-bit/8-bit activation quantization |
| **MTP (Multi-Token Prediction)** | Predict multiple future tokens |
| **Skip-gram Loss** | Auxiliary prediction heads |

### Training Features
| Feature | Description |
|---------|-------------|
| **Muon + AdamW** | Combined optimizer (orthogonal init + Adam) |
| **Gradient Checkpointing** | Memory-efficient backpropagation |
| **EMA** | Exponential moving average |
| **Curriculum Learning** | Progressive sequence length increase |
| **Auto SFT/DPO** | Automatic post-training fine-tuning |

---

## Quick Start

```bash
# Install dependencies
make install

# Download datasets
make data

# Build and tokenize
make build

# Train with AutoPilot (recommended)
python -m bulba1.cli --config configs/default.yaml --auto

# Train without AutoPilot
make train
```

### Resume Training

```bash
python -m bulba1.cli --config configs/default.yaml --resume checkpoints/run_bulba1_67m
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MiniChat (Main Model)                   │
├─────────────────────────────────────────────────────────────┤
│  Embedding → [Block × 10] → Norm → LM Head (TiedHead)       │
│                      ↓                                       │
│              ┌─────────────────────────────────┐             │
│              │         Block (10 layers)        │             │
│  ┌───────────┴───────────┬────────────────────┴──────────┐  │
│  │                       │                               │  │
│  │    ┌─────────────┐    │    ┌─────────────┐          │  │
│  │    │  MoDGate    │    │    │TokenMerger  │          │  │
│  │    │(Mixture of  │    │    │ (Inference) │          │  │
│  │    │   Depths)   │    │    │             │          │  │
│  │    └─────────────┘    │    └─────────────┘          │  │
│  │                       │                               │  │
│  │    ┌───────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │    │  DiffAttn │  │KDA+Mamba  │  │  MoE Layer │    │  │
│  │    │(MLA+RoPE) │  │   (SSM)   │  │ (4 experts)│    │  │
│  │    └───────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Model Configuration (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Embedding dimension |
| n_layers | 10 | Number of transformer layers |
| n_heads | 8 | Attention heads |
| vocab_size | 26000 | Vocabulary size |
| num_experts | 4 | MoE experts |
| top_k | 2 | Active experts per token |
| batch_size | 32 | Training batch size |
| seq_len | 512 | Sequence length |

**Effective Parameters**: ~26.7M (≈150-200M traditional transformer capacity with all optimizations)

---

## Configuration

### Basic Configuration (configs/default.yaml)

```yaml
model:
  d_model: 512
  n_layers: 10
  n_heads: 8
  vocab_size: 26000
  num_experts: 4
  top_k: 2

training:
  batch_size: 32
  seq_len: 512
  total_steps: 38035
  learning_rate: 0.0005
  use_gradient_checkpointing: true
  compile: false
```

### VRAM-Constrained (RTX 3060 12GB)

```yaml
model:
  d_model: 384
  n_layers: 8
  batch_size: 16

training:
  batch_size: 16
  seq_len: 256
  use_gradient_checkpointing: true
```

### Maximum Quality (RTX 4090+)

```yaml
model:
  d_model: 768
  n_layers: 16
  num_experts: 8
  batch_size: 16
  seq_len: 1024

training:
  use_gradient_checkpointing: true
  compile: true
```

---

## AutoPilot (Autonomous Training)

AutoPilot automatically adjusts hyperparameters during training using a state machine:

```
CALIBRATE → EXPLORE → EXPLOIT → PLATEAU → SGDR (warm restarts)
```

### Enable AutoPilot

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

### AutoPilot Features

- **Dynamic LR** — Computes optimal learning rate based on loss curvature
- **Plateau Detection** — Monitors loss stagnation, triggers recovery
- **Warm Restarts** — SGDR-style restarts for escaping local minima
- **Adaptive Weight Decay** — Adjusts regularization based on training progress
- **Gradient Noise Injection** — Dynamic noise for better generalization

---

## Auto SFT/DPO Pipeline

After main training completes, automatically run SFT (Supervised Fine-Tuning) and/or DPO (Direct Preference Optimization).

### Configuration

```yaml
training:
  # SFT Configuration
  auto_sft: true
  auto_sft_data: "data/sft"
  auto_sft_epochs: 3
  auto_sft_lr: 1.0e-5

  # DPO Configuration
  auto_dpo: true
  auto_dpo_data: "data/dpo"
  auto_dpo_epochs: 3
  auto_dpo_lr: 1.0e-6
  auto_dpo_beta: 0.1
```

### Data Format

**SFT** (`data/sft/train.jsonl`):
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is..."}]}
```

**DPO** (`data/dpo/train.jsonl`):
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is..."}]}
```

### Run SFT/DPO Manually

```bash
# SFT Training
python scripts/sft_train.py --data data/sft --output checkpoints/sft --epochs 3 --lr 1.0e-5

# DPO Training
python scripts/dpo_train.py --data data/dpo --output checkpoints/dpo --epochs 3 --lr 1.0e-6
```

---

## Training

### Command Line Options

```bash
python -m bulba1.cli [OPTIONS]

Options:
  --config PATH          Config file (default: configs/default.yaml)
  --auto                 Enable AutoPilot
  --resume PATH          Resume from checkpoint
  --data-dir PATH        Data directory
  --output-dir PATH      Output directory
```

### Monitoring

Training logs to `logs/bulba1.jsonl`:
```json
{"timestamp": "2026-05-16 14:20:14", "step": 400, "loss": 12.81, "lr": 1.6e-4, "vram_used_mb": 13734, "tok_per_sec": 1291}
```

### Checkpoints

Saved to `checkpoints/run_bulba1_67m/` with automatic rotation (keep top 3).

---

## VRAM Optimization

### VRAM Usage (RTX 5060 Ti 16GB)

| Config | Batch | Seq | VRAM | Notes |
|--------|-------|-----|------|-------|
| Default | 32 | 512 | ~12 GB | Stable |
| Large | 32 | 256 | ~10 GB | Fast |
| Constrained | 16 | 512 | ~8 GB | Safe |
| Minimal | 8 | 256 | ~5 GB | Very safe |

### OOM Protection

- Automatic batch size reduction on OOM
- VRAM threshold monitoring (warn at 88%, critical at 95%)
- Gradient checkpointing enabled by default
- Flash attention for memory efficiency

---

## Telegram Bot

Control training from Telegram:

### User Commands
- `/status` — Training status
- `/gpu` — GPU info
- `/logs` — Last log lines
- `/chat [text]` — Generate text
- `/help` — Show all commands

### Admin Commands
- `/train` — Start training
- `/stop` — Stop training
- `/restart` — Restart training

Configure bot token in `.env`:
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_ADMIN_IDS=your_id
```

---

## Project Structure

```
bulba1-python/
├── bulba1/
│   ├── model/
│   │   ├── minichat.py         # Main model (MiniChat, TiedHead)
│   │   ├── block.py            # Layer block with MoD
│   │   ├── kda.py              # Kimi Delta Attention
│   │   ├── diff_attn.py        # Differential attention
│   │   ├── mamba.py            # Mamba-3 SSM
│   │   ├── moe.py              # Mixture of Experts + ReX
│   │   ├── mhc.py              # Multi-head cache
│   │   ├── bit_linear.py       # BitNet quantization
│   │   ├── mod.py              # MoDGate (Mixture of Depths)
│   │   └── token_merging.py    # Token merging
│   ├── training/
│   │   ├── engine.py           # Training loop + AutoPilot
│   │   ├── optimizer.py        # Muon + AdamW
│   │   ├── ema.py              # Exponential moving average
│   │   ├── checkpoint.py       # Checkpoint management
│   │   ├── monitor.py          # VRAM/GPU monitoring
│   │   ├── eval.py             # Evaluation
│   │   └── stages.py           # Multi-stage training
│   ├── autonomy.py             # AutoPilot (autonomous HP)
│   ├── cli.py                  # Command-line interface
│   ├── orchestrator.py        # End-to-end pipeline
│   ├── tokenizer.py           # Custom tokenizer
│   └── config.py              # Model config
├── configs/
│   ├── default.yaml           # Main config (67M)
│   ├── auto.yaml              # AutoPilot config
│   └── smoke_test.yaml        # Quick test
├── scripts/
│   ├── download_all_datasets.py  # Download training data
│   ├── build_and_tokenize.py     # Build dataset
│   ├── sft_train.py              # SFT training
│   ├── dpo_train.py              # DPO training
│   └── pretokenize.py            # Pretokenize data
├── docs/
│   ├── ARCHITECTURE.md      # Architecture details
│   ├── COMPONENTS.md        # Component reference
│   ├── CONFIG_GUIDE.md      # Configuration guide
│   ├── TRAINING.md          # Training guide
│   └── DEVELOPER_GUIDE.md   # Developer guide
├── tools/
│   ├── deep_profile.py      # GPU profiling
│   └── log_viz.py           # Log visualization
├── data/
│   ├── sft/                 # SFT training data
│   ├── dpo/                 # DPO training data
│   └── tokenized/           # Tokenized dataset
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
└── tests/                   # Unit tests
```

---

## Requirements

- **Python**: 3.12+
- **CUDA**: 12.1+
- **GPU**: 12GB VRAM minimum (RTX 3060+)
- **RAM**: 16GB system memory
- **Disk**: 100GB+ for datasets

### Python Dependencies

```
torch>=2.1.0
transformers>=4.36.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
```

---

## Git

```
Codeberg: https://codeberg.org/sehaxe/bulba1
GitHub:   https://github.com/sehaxe/bulba1
```

---

## License

MIT License

---

## Acknowledgments

- Mamba architecture from [Mamba-SS](https://github.com/state-spaces/mamba)
- KDA from [Kimi](https://github.com/MoonshotAI/kimi-dev)
- MoE from [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)
- BitNet from [Microsoft](https://github.com/microsoft/BitNet)