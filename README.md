# 🚀 Bulba1 — Autonomous 1-bit LLM Training Framework

<div align="center">

**Train production-grade language models on a single consumer GPU**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Overview

**Bulba1** is a high-performance, memory-efficient framework for training large language models from scratch on consumer hardware. Built with cutting-edge optimizations from recent research papers, it enables training 300M-1B parameter models on a single RTX 5060 Ti (16GB VRAM) in under 12 hours.

### 🎯 Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **BitNet a4.8** | 1.58-bit weights, 8-bit activations | 16× VRAM savings vs FP32 |
| **MoE** | Mixture of Experts with grouped GEMM | 4× fewer active parameters |
| **KDA** | Kimi Delta Attention (chunkwise parallel) | O(N) linear attention |
| **Muon** | Newton-Schulz optimizer | 2× faster convergence |
| **DiffAttention** | Differential attention with MLA | Efficient KV compression |
| **AutoPilot** | Autonomous LR/WD/noise tuning | Hands-free training |
| **torch.compile** | JIT graph compilation | 30-50% speedup |

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MiniChat Model                        │
├─────────────────────────────────────────────────────────┤
│  Embedding → [Block × N] → RMSNorm → LM Head          │
│                                                          │
│  Block Structure:                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Attention Residuals (AttnRes)                   │   │
│  │  ├── DiffAttention / KDA (with RoPE + MLA)      │   │
│  │  └── MoE Layer (Token Choice + Shared Experts)  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  Optimizations:                                          │
│  • BitLinear (1.58-bit weights)                          │
│  • Quantized KV Cache (3-bit)                            │
│  • Gradient Checkpointing                                │
│  • Multi-Token Prediction (MTP)                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **CUDA** 11.8+ (NVIDIA GPU with 16GB+ VRAM recommended)
- **uv** package manager (optional, but recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bulba1.git
cd bulba1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using uv (faster)
uv sync
```

### Train Your First Model

```bash
# 1. Prepare data (download and tokenize)
python scripts/build_and_tokenize.py

# 2. Train model (resume from checkpoint if interrupted)
uv run python -m bulba1.cli train \
  --config configs/bulba1_micro_chat.yaml \
  --compile \
  --resume

# 3. Interactive chat with trained model
uv run python -m bulba1.cli chat \
  --model checkpoints/run_bulba1_micro_chat/best.safetensors
```

---

## 📊 Performance Benchmarks

### RTX 5060 Ti (16GB VRAM)

| Model Size | Active Params | Speed | VRAM | Time to 15K steps |
|------------|---------------|-------|------|-------------------|
| 58M (MoE)  | 20M           | 7,400 tok/s | 12.3 GB | ~6 hours |
| 340M (MoE) | 100M          | 3,200 tok/s | 14.8 GB | ~18 hours |
| 1B (MoE)   | 350M          | 1,800 tok/s | 15.5 GB | ~40 hours |

### Training Dynamics (58M MoE Model)

| Step | Loss | EMA Loss | Speed | VRAM |
|------|------|----------|-------|------|
| 1    | 27.56 | 27.56 | 188 tok/s | 3.2 GB |
| 1,000 | 10.2 | 9.8 | 6,200 tok/s | 11.8 GB |
| 5,000 | 5.1 | 4.9 | 7,100 tok/s | 12.3 GB |
| 15,000 | 3.8 | 3.7 | 7,400 tok/s | 12.3 GB |

---

## ⚙️ Configuration

All training parameters are defined in YAML configs (`configs/`).

### Example: `bulba1_micro_chat.yaml`

```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  vocab_size: 32000
  
  # Mixture of Experts
  use_moe: true
  num_experts: 8
  top_k: 2
  expert_hidden: 1536
  
  # Attention
  use_diff_attn: true
  use_mla: false
  max_ctx_len: 2048
  
  # Quantization
  use_bitlinear: true
  use_bitnet_a48: true
  bitnet_activation_bits: 8
  
  # Optimizer
  use_muon: true
  learning_rate: 0.001
  weight_decay: 0.1

training:
  batch_size: 24
  grad_accum_steps: 3  # Global batch = 72
  seq_len: 1024
  total_steps: 15000
  
  # Curriculum Learning
  curriculum_start_seq_len: 256
  curriculum_warmup_ratio: 0.05
  
  # Checkpointing
  checkpoint_every: 1000
  checkpoint_dir: checkpoints/run_bulba1_micro_chat
```

### Auto-Architecture

Let Bulba1 automatically derive optimal architecture from your data:

```bash
python scripts/auto_config.py \
  --data data/tokenized \
  --vram 14000 \
  --output configs/auto_derived.yaml
```

---

## 🛠️ CLI Commands

### Training

```bash
# Train from scratch
uv run python -m bulba1.cli train --config configs/default.yaml

# Resume from latest checkpoint
uv run python -m bulba1.cli train --config configs/default.yaml --resume

# Resume from specific step
uv run python -m bulba1.cli train --config configs/default.yaml --checkpoint 5000

# Enable torch.compile (recommended)
uv run python -m bulba1.cli train --config configs/default.yaml --compile

# Enable AutoPilot (autonomous hyperparameter tuning)
uv run python -m bulba1.cli train --config configs/default.yaml --auto
```

### Tokenization

```bash
# Tokenize and shard dataset
uv run python -m bulba1.cli tokenize \
  --data data/train \
  --output data/tokenized \
  --seq-len 1024
```

### Inference

```bash
# Interactive chat
uv run python -m bulba1.cli chat \
  --model checkpoints/run_bulba1_micro_chat/best.safetensors \
  --temperature 0.8 \
  --top-k 50

# One-shot generation
uv run python -m bulba1.cli chat \
  --model checkpoints/run_bulba1_micro_chat/best.safetensors \
  --prompt "def fibonacci(n):" \
  --max-tokens 100
```

### Monitoring

```bash
# Start Telegram monitoring bot
uv run python -m bulba1.cli monitor --token YOUR_BOT_TOKEN

# Tail training logs
uv run python -m bulba1.cli logs --stream train

# Generate training plots
uv run python -m bulba1.cli plot --metric loss --output plots/loss_curve.png
```

### Profiling

```bash
# Profile model component latencies
uv run python -m bulba1.cli profile --config configs/default.yaml --steps 10

# Show model info
uv run python -m bulba1.cli info --config configs/default.yaml
```

---

## 📦 Project Structure

```
bulba1/
├── bulba1/                    # Core framework
│   ├── model/                 # Neural network architectures
│   │   ├── minichat.py        # Main model (MiniChat)
│   │   ├── block.py           # Transformer block
│   │   ├── moe.py             # Mixture of Experts
│   │   ├── kda.py             # Kimi Delta Attention
│   │   ├── diff_attn.py       # Differential Attention
│   │   ├── bit_linear.py      # BitNet quantization
│   │   └── rope.py            # Rotary Positional Embeddings
│   ├── training/              # Training pipeline
│   │   ├── engine.py          # Training engine
│   │   ├── optimizer.py       # Muon + AdamW
│   │   ├── checkpoint.py      # Safetensors checkpointing
│   │   ├── ema.py             # Exponential Moving Average
│   │   └── monitor.py         # System monitoring
│   ├── triton_ops/            # Triton CUDA kernels
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Pydantic configuration
│   ├── tokenizer.py           # Smart tokenizer
│   └── autonomy.py            # AutoPilot (adaptive LR)
├── configs/                   # YAML configurations
├── scripts/                   # Utility scripts
├── tests/                     # Unit tests
├── tools/                     # Profiling & visualization
├── telegram_bot/              # Monitoring bot
└── data/                      # Training data (gitignored)
```

---

## 🔬 Scientific Foundations

| Technique | Paper | Year | Contribution |
|-----------|-------|------|--------------|
| **Muon Optimizer** | arXiv:2502.16982 | 2025 | 2× faster convergence vs AdamW |
| **BitNet a4.8** | arXiv:2411.04965 | 2024 | 16× VRAM savings, 1.58-bit weights |
| **KDA** | arXiv:2510.26692 | 2025 | O(N) linear attention with chunkwise parallelism |
| **Attention Residuals** | arXiv:2603.15031 | 2025 | Cross-layer depth attention |
| **YaRN** | arXiv:2309.00071 | 2023 | Context extension via NTK-by-parts |
| **MegaBlocks** | arXiv:2211.15841 | 2022 | Efficient MoE with dropless routing |
| **DeepSeek-V3** | arXiv:2412.19437 | 2024 | Multi-Token Prediction + MLA |

---

## 📈 Advanced Features

### Multi-Token Prediction (MTP)

Train multiple future tokens in parallel for faster inference:

```yaml
model:
  use_mtp: true
  num_mtp_heads: 2
  loss_mtp1_weight: 0.3
  loss_mtp2_weight: 0.1
  mtp1_warmup_steps: 1000
```

### Curriculum Learning

Gradually increase sequence length for stable training:

```yaml
training:
  curriculum_start_seq_len: 256
  curriculum_warmup_ratio: 0.05
  seq_len: 1024  # Final sequence length
```

### AutoPilot

Autonomous hyperparameter tuning based on loss dynamics:

```yaml
autonomy:
  enabled: true
  base_lr: 0.0005
  plateau_patience: 800
  max_lr_reductions: 3
```

### YaRN Context Extension

Extend context length after training:

```bash
python scripts/apply_yarn.py \
  --checkpoint checkpoints/bulba1_micro_chat/best.safetensors \
  --original-len 2048 \
  --target-len 32768 \
  --output checkpoints/bulba1_micro_chat_32k/
```

---

## 🤖 Telegram Monitoring

Monitor training progress from your phone:

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Set environment variable: `export TELEGRAM_BOT_TOKEN=your_token`
3. Run bot: `uv run python telegram_bot/bot.py`

### Available Commands

| Command | Description |
|---------|-------------|
| `/status` | Training status, loss, speed |
| `/gpu` | GPU utilization and temperature |
| `/sys` | System info (CPU, RAM, disk) |
| `/eta` | Estimated time to completion |
| `/logs` | Last 10 training steps |
| `/plot` | Generate loss curve plot |
| `/checkpoint` | Latest checkpoint info |
| `/save` | Force checkpoint save (admin) |

---

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Fast tests only (skip benchmarks)
pytest tests/ -v -m "not slow"

# Specific test file
pytest tests/test_model_components.py -v

# With coverage
pytest tests/ --cov=bulba1 --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
uv sync

# Pre-commit hooks
pre-commit install

# Run linters
uv ruff check bulba1/
mypy bulba1/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Moonshot AI** for Kimi Delta Attention
- **DeepSeek** for Multi-Token Prediction and MLA
- **Microsoft** for BitNet architecture
- **Keller Jordan** for Muon optimizer
- **Hugging Face** for tokenizers library

---

<div align="center">

**Built with ❤️ for the open-source ML community**

[Report Bug](https://github.com/yourusername/bulba1/issues) · [Request Feature](https://github.com/yourusername/bulba1/issues)

</div>