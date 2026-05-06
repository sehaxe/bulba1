# Bulba 1 «Singularity»

Autonomous LLM training platform in pure Python + PyTorch. 225M params, 16 layers, trained on consumer hardware (RTX 5060 Ti 16GB).

## Требования

### Аппаратные
- **GPU**: NVIDIA с CUDA 13.0+ и bf16 support
- **VRAM**: 16GB (тестировано на RTX 5060 Ti)
- **RAM**: 16GB system
- **Storage**: 10GB для данных + чекпоинтов

### Программные
- Python 3.11+
- PyTorch 2.11+ с CUDA
- CUDA 13.0

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/your-repo/bulba1-python.git
cd bulba1-python

# 2. Установить зависимости
uv sync

# 3. Создать директорию данных
mkdir -p data/train

# 4. Запустить обучение (из корня проекта!)
./bulba --params 225M --steps 1000
# или
python bulba --params 225M --steps 1000
```

## Установка

### 1. Установка Python и uv

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv curl

# Установить uv (менеджер пакетов)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Или через pip
pip install uv
```

### 2. Клонирование и установка зависимостей

```bash
git clone https://github.com/your-repo/bulba1-python.git
cd bulba1-python
uv sync
```

Это автоматически установит:
- torch>=2.11.0 (с CUDA 13.0)
- tokenizers>=0.23.0
- datasets>=4.8.0
- rich, textual, tensorboard
- bitsandbytes (для 8-bit optimizer)
- matplotlib, pandas (для визуализации)

### 3. Проверка GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести:
```
CUDA: True, GPU: NVIDIA GeForce RTX 5060 Ti
```

## Подготовка данных

### Структура данных

```
data/
├── tokenizer_fast.json    # Токенизатор (автоматически создается)
└── train/
    ├── fineweb-000.txt    # Web data (~3GB)
    ├── c4.txt             # Common Crawl (~1GB)
    ├── codeparrot.txt     # Python code (~600MB)
    ├── code_search_net.txt
    ├── arxiv_markdown.txt
    └── ...
```

### Скачивание данных

```bash
# Создать директорию
mkdir -p data/train

# Пример: скачать часть FineWeb (10B tokens)
# Внимание: это ~50GB, используй только нужное
python -c "
import requests
import os
os.makedirs('data/train', exist_ok=True)
url = 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-10b/resolve/main/data/CC-MAIN-2024-18/'
# Скачай нужные файлы через huggingface-cli или wget
"

# Или используй готовые датасеты
# C4: https://huggingface.co/datasets/naver/clueweb22-bm-pt
```

### Токенизация

Токенизатор создаётся автоматически при первом запуске:
```bash
python bulba1/cli.py --params 225M --steps 10
# Создаст data/tokenizer_fast.json
```

## Обучение

### Основной скрипт (225M params, 16 layers)

```bash
python run_132m_12l.py
```

Это использует конфигурацию:
```python
cfg = ModelConfig(
    d_model=768, n_layers=16, n_heads=12,
    num_experts=16, expert_hidden=768,
    vocab_size=12000,
    batch_size=5, seq_len=512,
    learning_rate=2e-4,
    use_mamba=True, use_bitlinear=True,
    use_rex=True, use_mtp=True,
    use_gradient_checkpointing=True,
)
```

### Через CLI

```bash
# Обучить 225M модель
python bulba1/cli.py --params 225M --steps 100000 --batch-size 5 --seq-len 512

# Свои параметры
python bulba1/cli.py --params 131M --steps 50000 --batch-size 4 --seq-len 256

# Продолжить обучение
python bulba1/cli.py --params 225M --steps 50000 --resume
```

### Основные параметры CLI

| Параметр | Описание | По умолчанию |
|----------|-----------|---------------|
| `--params` | Размер модели (125M, 1B, etc) | 225M |
| `--steps` | Количество шагов | 1000 |
| `--batch-size` | Размер батча | auto |
| `--seq-len` | Длина последовательности | 128 |
| `--lr` | Learning rate | 3e-4 |
| `--compile` | Использовать torch.compile | off |
| `--resume` | Продолжить с чекпоинта | off |
| `--data-dir` | Папка с данными | data/train |

### Systemd сервисы (рекомендуется)

Сервисы находятся в `services/`:

```bash
# Скопировать в ~/.config/systemd/user/
mkdir -p ~/.config/systemd/user
cp services/*.service ~/.config/systemd/user/

# Перезагрузить демон
systemctl --user daemon-reload

# Включить автозапуск (опционально)
systemctl --user enable bulba1-225m.service
systemctl --user enable bulba1-bot.service

# Запустить
systemctl --user start bulba1-225m     # Training
systemctl --user start bulba1-bot       # Telegram bot

# Проверить статус
systemctl --user status bulba1-225m
systemctl --user status bulba1-bot

# Смотреть логи
journalctl --user -u bulba1-225m -f
journalctl --user -u bulba1-bot -f
```

**Требования для сервисов:**
- Запусти `uv sync` в проекте хотя бы раз (создаст .venv)
- Настрой токен бота в `telegram_bot/bot_config.py`

## Telegram бот

### Установка токена

1. Создай бота через @BotFather в Telegram
2. Получи токен
3. Создай файл `.env`:
```bash
echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env
echo "ADMIN_USER_ID=your_telegram_id" >> .env
```

### Запуск бота

```bash
# Вручную
python telegram_bot/bot.py

# Через systemd
systemctl --user enable bulba1-bot.service
systemctl --user start bulba1-bot
```

### Команды бота

**Пользовательские:**
- `/status` — статус обучения
- `/gpu` — информация о GPU
- `/system` — CPU/RAM/Disk
- `/logs` — последние 20 строк лога
- `/plot` — график loss
- `/eta` — ETA завершения
- `/chat [текст]` — сгенерировать текст (3 раза/день)

**Админские:**
- `/train` — начать обучение
- `/stop` — остановить (с подтверждением)
- `/restart` — перезапустить
- `/reset` — сбросить чекпоинты (с подтверждением)
- `/quit` — выйти из сессии

### Настройка админа

В файле `telegram_bot/bot.py` измени:
```python
ADMIN_USER_IDS = {5554531940}  # твой Telegram ID
```

Узнать свой ID: @userinfobot

## Чекпоинты и экспорт

### Сохранение

Чекпоинты сохраняются в `checkpoints/run_bulba1_225m_clean/` каждые 100 шагов.

### Экспорт в HuggingFace формат

```bash
python bulba1/cli.py --export-hf --checkpoint-dir checkpoints/run_bulba1_225m_clean
```

Это создаст:
- `hf_export/model.pt` — веса модели
- `hf_export/config.json` — конфиг

### Продолжение обучения

```bash
python bulba1/cli.py --resume --checkpoint-dir checkpoints/run_bulba1_225m_clean
```

## Мониторинг

### Логи

```bash
# Systemd
journalctl --user -u bulba1 -f

# Файл
tail -f logs/bulba1_225m.log
```

### Графики

```bash
# С помощью matplotlib
python tools/log_viz.py logs/bulba1_225m.log
```

### Мониторинг GPU

```bash
# В реальном времени
watch -n 1 nvidia-smi

# Через бота
/gpu
```

## Устранение проблем

### OOM (Out of Memory)

```bash
# Уменьшить batch_size
python bulba1/cli.py --params 225M --batch-size 3

# Включить gradient checkpointing (автоматически)
# Уменьшить seq_len
python bulba1/cli.py --params 225M --seq-len 256
```

### CUDA out of memory при генерации

Бот автоматически использует CPU для генерации когда идёт обучение.

### Чекпоинты не сохраняются

Проверь права на запись:
```bash
ls -la checkpoints/
mkdir -p checkpoints/run_bulba1_225m_clean
```

### Обучение не начинается

Проверь данные:
```bash
ls data/train/*.txt | head
# Должны быть .txt файлы
```

### Бот не отвечает

```bash
# Проверь токен
echo $TELEGRAM_BOT_TOKEN

# Перезапусти
systemctl --user restart bulba1-bot
```

## Конфигурация модели

Текущая конфигурация (225M params):

```python
# Архитектура
d_model = 768          # Размер эмбеддинга
n_layers = 16          # Количество слоев
n_heads = 12           # Attention heads
num_experts = 16       # MoE experts
expert_hidden = 768   # Hidden size в experts
vocab_size = 12000    # Размер словаря

# Обучение
batch_size = 5
seq_len = 512
learning_rate = 2e-4
weight_decay = 0.1

# Оптимизации
use_mamba = True           # SSM слои
use_bitlinear = True       # Ternary weights
use_rex = True            # ReX (reuse previous layer)
use_mtp = True            # Multi-Token Prediction
use_gradient_checkpointing = True
use_f16 = True            # BF16 mixed precision
```

### VRAM usage

| Конфигурация | VRAM |
|--------------|------|
| batch=5, seq=512 | ~14 GB |
| batch=4, seq=512 | ~12 GB |
| batch=3, seq=256 | ~8 GB |

## Архитектура

Bulba 1 использует гибридную архитектуру:

- **DiffAttention** — дифференциальное внимание с RoPE и QK-Norm
- **MoE + ReX** — 16 экспертов с top-k routing и reuse из предыдущего слоя
- **Mamba-2 SSD** — state space модель для длинных зависимостей
- **BitNet** — ternary quantization весов (-1, 0, +1)
- **MTP** — multi-token prediction для предсказания нескольких токенов вперёд
- **CLR** — learnable tokens для рассуждений

Подробнее: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Структура проекта

```
bulba1-python/
├── bulba.py                    # Entry point
├── run_132m_12l.py             # Main training script
├── pyproject.toml              # Dependencies
│
├── bulba1/                     # Main package
│   ├── cli.py                  # CLI interface
│   ├── model/
│   │   ├── minichat.py         # Full model
│   │   ├── block.py            # Transformer block
│   │   ├── diff_attn.py       # Differential attention
│   │   ├── moe.py             # MoE + ReX
│   │   ├── mamba.py           # Mamba-2 SSD
│   │   ├── bit_linear.py      # BitNet quantization
│   │   ├── kda.py             # Kimi Delta Attention
│   │   └── mhc.py             # Hyper-connections
│   ├── training/
│   │   ├── engine.py          # Training loop
│   │   ├── optimizer.py       # AdamW / Muon
│   │   ├── checkpoint.py     # Save/load
│   │   ├── autotuner.py      # Auto hardware config
│   │   └── ema.py            # Exponential moving avg
│   └── data/
│       └── tokenizer.py       # Tokenizer
│
├── telegram_bot/
│   └── bot.py                 # Telegram monitoring bot
│
├── tools/
│   ├── log_viz.py            # Plot training curves
│   └── benchmark.py          # Performance benchmark
│
├── docs/
│   ├── ARCHITECTURE.md       # Model architecture details
│   ├── DEVELOPER_GUIDE.md    # For contributors
│   └── PAPERS.md            # Research papers
│
├── data/
│   ├── tokenizer_fast.json   # Trained tokenizer
│   └── train/               # Training data (.txt files)
│
└── checkpoints/             # Model checkpoints
```

## Документация

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Полное описание архитектуры модели
- [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — Гайд для разработчиков

## Лицензия

MIT

## TODO

- [ ] Добавить интеграцию с Weights & Biases
- [ ] Добавить FSDP для multi-GPU
- [ ] Добавить LoRA fine-tuning