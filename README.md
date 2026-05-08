# ⚡ Bulba 1 — Autonomous LLM Training

Self-hosted LLM training on consumer GPU. **225M params, 16 layers** — runs on RTX 5060 Ti 16GB.

## 🚀 Быстрый старт

git clone <repo>
cd bulba1-python

make install          # установить зависимости (Mamba-3 wheel + dev)
make data             # скачать датасеты
make build            # собрать датасет и обучить токенизатор
make install-services # установить systemd-юниты

systemctl --user start bulba1   # запустить обучение
journalctl --user -u bulba1 -f  # смотреть логи

## 🚀 Features

| Feature | Description |
|--------|------------|
| **Hybrid Architecture** | 75% Mamba-2 SSD + 25% KDA + MoE |
| **BitNet** | Ternary quantization (-1, 0, +1) |
| **Smart Checkpoints** | 100/500/1000 steps (adaptive) |
| **Telegram Bot** | Monitor training, generate text |
| **Zero OOM** | Auto VRAM management |

## 📊 Architecture

```
Layer 0:   KDA + MoE      (attention)
Layer 1-3: Mamba-2 SSD
Layer 4:   KDA + MoE
Layer 5-7: Mamba-2 SSD
...
```

- **Mamba-2 SSD**: Linear complexity, no KV cache
- **KDA**: Kimi Delta Attention (75% less KV)
- **MoE**: 16 experts, top-k routing
- **ReX**: Expert reuse from previous layer
- **MTP**: Multi-token prediction (t+1, t+2)

## 💾 VRAM Usage

| batch | seq | VRAM |
|-------|-----|------|
| 5 | 512 | ~14 GB |
| 4 | 512 | ~12 GB |
| 3 | 256 | ~8 GB |

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
uv run python run_225m.py

# Or via systemd
systemctl --user start bulba1-225m     # Training
systemctl --user start bulba1-bot       # Bot

# Check status
journalctl --user -u bulba1-225m -f
```

## 📁 Structure

```
bulba1/
├── run_225m.py          # Main training script
├── services/           # Systemd services
├── bulba1/
│   ├── model/         # Architecture (10 files)
│   ├── training/      # Engine (12 files)
│   └── cli.py         # CLI
├── telegram_bot/       # Monitoring bot
└── tools/             # Utilities
```

## 🤖 Telegram Bot

- `/status` — Training status
- `/gpu` — GPU info
- `/logs` — Last log lines
- `/plot` — Loss graph
- `/chat [text]` — Generate text

**Admin**: `/train`, `/stop`, `/restart`

## 📝 Config (run_225m.py)

```python
d_model = 768
n_layers = 16
n_heads = 12
num_experts = 16
batch_size = 5
seq_len = 512
learning_rate = 2e-4

use_mamba = True
use_bitlinear = True
use_rex = True
use_mtp = True
```

**Git**: https://codeberg.org/quazder/bulba1