# ⚡ Bulba 1 — Autonomous LLM Training

Self-hosted LLM training on consumer GPU. **340M params, 16 layers** — runs on RTX 5060 Ti 16GB.

## 🚀 Быстрый старт

```bash
git clone <repo>
cd bulba1-python

make install          # установить зависимости
make data             # скачать датасеты
make build            # собрать датасет и обучить токенизатор
make install-services # установить systemd-юниты

systemctl --user start bulba1   # запустить обучение
journalctl --user -u bulba1 -f  # смотреть логи
```

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | Mamba-2 SSD + KDA + MoE |
| **MHC** | Multi-head Latent Clustering (DeepSeek style) |
| **Muon + AdamW** | Custom optimizer with Newton-Schulz |
| **Gradient Checkpointing** | Memory-efficient training |
| **Telegram Bot** | Monitor training, generate text |
| **Smart Checkpoints** | Auto-save with rotation |

## 📊 Architecture

```
Layer 0,4,8,12:   KDA + MoE      (attention blocks)
Layer 1-3,5-7:    Mamba-2 SSD
...
```

- **Mamba-2 SSD**: Linear complexity, no KV cache
- **KDA**: Kimi Delta Attention (reduced memory)
- **MoE**: 16 experts, top-k routing with ReX
- **MTP**: Multi-token prediction (t+1, t+2)
- **MHC**: Latent clustering for better representations

## 💾 VRAM Usage (RTX 5060 Ti 16GB)

| batch | seq | VRAM |
|-------|-----|------|
| 28 | 512 | ~14 GB |
| 32 | 256 | ~12 GB |
| 16 | 512 | ~8 GB |

## 🛠 Setup

```bash
# Install dependencies
uv sync

# Configure Telegram bot
cp telegram_bot/bot_config.py.example telegram_bot/bot_config.py
# Edit with your bot token and admin ID

# Install systemd services
cp services/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

## ▶️ Usage

```bash
# Direct training
make train

# Or via systemd
systemctl --user start bulba1        # Training
systemctl --user start bulba1-bot    # Bot

# Check logs
journalctl --user -u bulba1 -f
```

## 📁 Project Structure

```
bulba1-python/
├── configs/
│   └── default.yaml       # Main training config
├── bulba1/
│   ├── model/            # Model architecture
│   │   ├── minichat.py   # Main model
│   │   ├── block.py      # Transformer block
│   │   ├── mamba.py     # Mamba-2 SSD
│   │   ├── kda.py        # Kimi Delta Attention
│   │   ├── moe.py        # Mixture of Experts
│   │   ├── mhc.py        # Multi-head Latent Clustering
│   │   └── bit_linear.py # BitNet quantization
│   ├── training/
│   │   ├── engine.py     # Training loop
│   │   ├── optimizer.py  # Muon + AdamW
│   │   ├── ema.py        # EMA weights
│   │   └── checkpoint.py # Smart checkpoints
│   ├── tokenizer.py      # Tokenizer
│   └── cli.py            # CLI interface
├── telegram_bot/          # Monitoring bot
├── tools/                # Profiling tools
└── services/             # Systemd services
```

## 🤖 Telegram Bot

- `/status` — Training status
- `/gpu` — GPU info  
- `/logs` — Last log lines
- `/chat [text]` — Generate text

**Admin**: `/train`, `/stop`, `/restart`

## 📝 Configuration

Edit `configs/default.yaml` to customize:

```yaml
training:
  batch_size: 28
  seq_len: 512
  learning_rate: 0.0015
  log_every: 50
  early_log_steps: 100
  early_log_every: 10

model:
  d_model: 768
  n_layers: 16
  num_experts: 16
  use_mhc: true
  mhc_n: 4
  use_muon: true
```

## 🔧 Profiling

```bash
# Deep profiling
uv run python tools/deep_profile.py

# Memory test
uv run python tools/memory_test.py
```

## 📈 Training Progress

Check progress via bot or logs:
```bash
tail -f logs/bulba1.jsonl
```

## ⚙️ Requirements

- Python 3.12+
- CUDA 12.1+
- 16GB VRAM (RTX 5060 Ti)
- 32GB RAM

**Git**: https://codeberg.org/quazder/bulba1