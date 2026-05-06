# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-07
**Type:** ML Training Platform (Pure Python + PyTorch)

## OVERVIEW

Autonomous LLM training in pure Python. 225M params, 16 layers. GPU: RTX 5060 Ti 16GB.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Run training | `run_132m_12l.py` | Main entry, 225M config |
| CLI | `bulba1/cli.py` | Arguments, exports |
| Model | `bulba1/model/` | 10 files - architecture |
| Training loop | `bulba1/training/engine.py` | Training logic |
| Data | `data/train/*.txt` | Training text files |
| Checkpoints | `checkpoints/` | Saved weights |
| Telegram bot | `telegram_bot/bot.py` | Monitoring |

## CONVENTIONS

- Python 3.11+ (pyproject.toml)
- Linting: Ruff (line-length=100, target=py311)
- Type: MyPy strict (disallow_untyped_defs)
- No .github/workflows / Makefile (not a CI project)

## COMMANDS

```bash
# Train 225M model
python run_132m_12l.py

# CLI
python bulba -p 225M -s 100000

# Telegram
python telegram_bot/bot.py
```

## NOTES

- Requires CUDA 13.0+ GPU with bf16 support
- VRAM: ~14GB @ batch=5, seq=512