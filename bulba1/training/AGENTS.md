# bulba1/training/ — Training Logic

**Score:** 18 (high complexity)

## OVERVIEW

12 files: engine, optimizer, checkpoint, eval, stages, LoRA, RLVR.

## FILES

| File | Role |
|------|------|
| `engine.py` | Training loop, TrainState |
| `optimizer.py` | AdamW, Muon |
| `checkpoint.py` | Save/load |
| `eval.py` | Evaluation |
| `stages.py` | Training stages |
| `lora.py` | LoRA fine-tuning |
| `rlvr.py` | RLVR scheduler |
| `autotuner.py` | Auto hardware config |
| `ema.py` | Exponential moving avg |
| `monitor.py` | Metrics |
| `chunked_ce.py` | Chunked cross-entropy |
| `__init__.py` | Exports |

## CONVENTIONS

- TrainState holds all training state
- stage-based training via `stages.py`
- Checkpoints every 100 steps (configurable)

## NOTES

- GPU required (CUDA)
- Mixed precision: bf16

## ANTI-PATTERNS (THIS PROJECT)

- Don't bypass TrainState for training state