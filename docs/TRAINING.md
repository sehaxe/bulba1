# Training Guide

This guide covers the training pipeline in Bulba 1, from data preparation to checkpointing.

---

## Quick Start

```bash
# Full pipeline
make install
make data
make build
make train

# Or manually
uv sync
python scripts/download_all_datasets.py
python scripts/build_and_tokenize.py

# Normal training
python -m bulba1.cli --config configs/default.yaml

# With AutoPilot (autonomous HP tuning)
python -m bulba1.cli --config configs/default.yaml --auto
```

---

## Training Pipeline Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Training Pipeline                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ Download│ →  │ Build   │ →  │ Tokenize│ →  │ Train   │ │
│  │  Data   │    │ Dataset │    │         │    │         │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                            │
│  Download:    scripts/download_all_datasets.py           │
│  Build:       scripts/build_and_tokenize.py              │
│  Train:       bulba1.cli (with/without --auto)            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## AutoPilot (Autonomous Training)

**NEW in v2:** AutoPilot automatically adjusts hyperparameters during training.

### Enable AutoPilot

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

### How it works

```python
# In TrainingEngine
if auto_mode:
    self.autopilot = AutoPilot(self.cfg, log_dir=self.cfg.log_dir)

# LR computation
def compute_lr(self, step, total):
    if self.autopilot is not None:
        return self.autopilot.compute_lr(step)  # Dynamic LR
    return cosine_decay(step, total)  # Default

# Step adjustment
action = self.autopilot.step(step, self.ema_loss)
# Adjusts: LR, weight decay, gradient noise, stochastic depth
```

### AutoPilot Modes

- **CALIBRATE**: Find stable LR
- **EXPLORE**: Try different hyperparameters
- **EXPLOIT**: Stick with best found
- **PLATEAU**: Detect plateau, trigger restart
- **SGDR**: Cyclic LR

### AutoPilot State

```python
# Check state
print(f"Mode: {engine.autopilot.state.mode}")
print(f"LR: {engine.autopilot.current_lr}")
print(f"WD: {engine.autopilot.current_wd}")
```

---

## Training Loop

### Normal Mode (without AutoPilot)

```python
engine = TrainingEngine(model, cfg, tokenizer, device="cuda", auto_mode=False)

for step in range(total_steps):
    # 1. Get batch
    batch = get_batch(engine, step)
    
    # 2. Forward pass
    logits, mtp1, mtp2, aux_loss = model(batch.input_ids)
    
    # 3. Compute loss
    loss = compute_loss(logits, mtp1, mtp2, aux_loss, labels)
    
    # 4. Backward + optimizer step
    loss.backward()
    grad_norm = engine._optimizer_step(step, total_steps)
    
    # 5. LR schedule (cosine)
    lr = engine.compute_lr(step, total_steps)
    engine.optimizer.set_lr(lr)
    
    # 6. Log & checkpoint
    if step % log_every == 0:
        log_metrics(step, loss, grad_norm, lr)
```

### AutoPilot Mode

```python
engine = TrainingEngine(model, cfg, tokenizer, device="cuda", auto_mode=True)

for step in range(total_steps):
    batch = get_batch(engine, step)
    logits, mtp1, mtp2, aux_loss = model(batch.input_ids)
    loss = compute_loss(logits, mtp1, mtp2, aux_loss, labels)
    loss.backward()
    grad_norm = engine._optimizer_step(step, total_steps)
    
    # AutoPilot LR (instead of cosine)
    lr = engine.compute_lr(step, total_steps)  # Calls autopilot.compute_lr()
    engine.optimizer.set_lr(lr)
    
    # AutoPilot step adjustment
    if step % 100 == 0:
        action = engine.autopilot.step(step, engine.ema_loss)
        # Adjusts: LR, WD, gradient noise, SD
    
    # Checkpoint includes autopilot state
    if step % checkpoint_every == 0:
        save_checkpoint(step, include_autopilot=True)
```

---

## Curriculum Learning

Dynamic sequence length based on training progress:

```yaml
curriculum_warmup_ratio: 0.15
curriculum_start_seq_len: 128
```

Default curriculum:
| Progress | Sequence Length |
|----------|-----------------|
| 0-15% | 128 |
| 15-100% | Gradual ramp to 512 |

---

## Checkpoint Management

```python
# Checkpoint includes:
{
    "step": 1000,
    "model_state": {...},
    "optimizer_state": {...},
    "config": {...},
    "autopilot": {...}  # AutoPilot state (if enabled)
}
```

### Resume with AutoPilot

```bash
python -m bulba1.cli --resume --checkpoint-dir checkpoints/run_bulba1_67m --auto
```

AutoPilot state is automatically restored to continue autonomous tuning.

---

## Loss Computation

```python
# Main loss
main_loss = cross_entropy(logits, labels)

# MTP losses (if enabled)
mtp_loss = cfg.loss_mtp1_weight * cross_entropy(mtp1, labels)
mtp_loss += cfg.loss_mtp2_weight * cross_entropy(mtp2, labels)

# Router auxiliary losses
aux_loss = router_z_loss + router_entropy_loss

# Total
total = main_loss + mtp_loss + aux_loss
```

---

## VRAM Management

### Automatic Batch Reduction

```python
if vram_pct > 95:  # Critical
    batch_size //= 2
    max 6 reductions
```

**Config:**
```yaml
vram_warn_pct: 88.0
vram_critical_pct: 95.0
max_batch_reductions: 6
```

### Memory Optimization Techniques

| Technique | VRAM Saved |
|-----------|------------|
| Gradient Checkpointing | ~30% |
| BF16 AMP | ~50% |
| Tied Embeddings | ~10% |
| Token Merging (inference) | ~30% |

---

## Monitoring

### Logs

```python
# logs/bulba1.jsonl
{
    "step": 1000,
    "loss": 2.45,
    "lr": 0.0005,
    "grad_norm": 1.2,
    "vram_gb": 10.5,
    "tokens_per_sec": 50.0,
    "batch_size": 32,
    "seq_len": 512,
    "autopilot_mode": "EXPLOIT"  # If --auto enabled
}
```

### AutoPilot Logs

```python
# logs/autonomy.jsonl
{
    "step": 1000,
    "mode": "EXPLOIT",
    "lr": 0.0005,
    "wd": 0.1,
    "noise": 3e-5,
    "action": "CONTINUE"
}
```

---

## CLI Options

```bash
python -m bulba1.cli --help

# Training
--config CONFIG          # Config file (default: configs/default.yaml)
--device DEVICE          # Device (default: auto)
--compile               # Use torch.compile
--auto                  # Enable AutoPilot (NEW)
--resume                # Resume from checkpoint

# Generation
--generate              # Run generation after training
--prompt PROMPT         # Prompt for generation
--max-new-tokens N      # Max tokens to generate

# Evaluation
--eval-every N          # Eval every N steps
--eval-prompts P1 P2    # Prompts for eval
```

---

## Troubleshooting

### Loss not converging

```bash
# Use AutoPilot for automatic adjustment
python -m bulba1.cli --config configs/default.yaml --auto
```

### OOM Errors

```yaml
# Reduce batch
batch_size: 16

# Enable gradient checkpointing
use_gradient_checkpointing: true

# Reduce sequence
seq_len: 256

# Disable Mamba
use_mamba: false
```

### Training unstable

```yaml
# Enable MHC
use_mhc: true

# Enable EMA
ema_decay: 0.999

# Reduce dropout
dropout: 0.0
```

---

## Performance

With default config (512d, 10 layers, 32 batch):

| Hardware | VRAM | Speed | Tokens/s |
|----------|------|-------|----------|
| RTX 5060 Ti 16GB | ~12 GB | 1.0 st/s | 50-60 |
| RTX 4090 24GB | ~18 GB | 1.5 st/s | 80-100 |