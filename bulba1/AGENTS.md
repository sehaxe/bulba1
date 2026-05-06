# bulba1/ — Main Package

**Score:** 18 (high complexity)

## OVERVIEW

Core ML training package: CLI, model architecture, training loop, utilities.

## STRUCTURE

```
bulba1/
├── cli.py           # Entry point, 18KB
├── orchestrator.py  # Training orchestration
├── model/         # 10 architecture files
├── training/      # 12 training files
├── data/         # Tokenizer
└── utils/        # Helpers
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Model config | `cli.py` | ModelConfig, TrainingConfig |
| Full model | `model/minichat.py` | Bulba1Chat model |
| Training loop | `training/engine.py` | TrainState, step() |
| Checkpointing | `training/checkpoint.py` | save/load |
| Optimizer | `training/optimizer.py` | AdamW, Muon |

## CONVENTIONS

- All model code in `model/`, training code in `training/`
- No cross-imports between model/training (orchestrator bridges)
- `__init__.py` exports nothing (explicit imports)

## ANTI-PATTERNS (THIS PROJECT)

- No cross-imports between model/ and training/ (orchestrator bridges)
- `__init__.py` exports nothing (explicit imports)