# Bulba 1 Architecture

## Overview

Bulba 1 — автономная платформа для обучения LLM на потребительском железе. Архитектура оптимизирована для 16GB VRAM (RTX 5060 Ti).

## Model Configuration

```
┌─────────────────────────────────────────────────────────────┐
│ Bulba 1 (225M params, 16 layers)                          │
├─────────────────────────────────────────────────────────────┤
│ d_model = 768          # Embedding size                    │
│ n_layers = 16          # Transformer layers                │
│ n_heads = 12           # Attention heads                    │
│ num_experts = 16       # MoE experts                        │
│ expert_hidden = 768    # Expert FFN hidden size            │
│ vocab_size = 12000     # Vocabulary                         │
│ seq_len = 512          # Maximum sequence length           │
│ batch_size = 5         # Training batch                    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Embedding

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

Токени → векторы размерности `d_model`.

### 2. Transformer Blocks (16 layers)

Каждый блок содержит:

```
Input → [DiffAttention + MoE + Mamba] → Output
```

#### 2.1 Differential Attention (DiffAttn)

```python
# Two parallel attention computations
out1 = Attention(Q, K, V)      # Main attention
out2 = Attention(Q, K2, V2)     # Delta attention
output = out1 - out2            # Differential output
```

**Особенности:**
- Latent KV compression (сжатие ключей/значений)
- RoPE (Rotary Position Embedding)
- QK-Norm (нормализация запросов/ключей)
- Per-head gating (ворота на каждую голову)

#### 2.2 MoE + ReX (Mixture of Experts)

```python
# Shared experts - всегда активны
shared_output = sum(shared_experts(x))

# Routed experts - top-k активация
routed_output = top_k routing(x) → experts

# ReX - reuse from previous layer
rex_output = prev_layer_experts(x)  # no_grad, no extra memory
```

**Особенности:**
- 16 experts (8 shared + 8 routed)
- Top-k=2 routing
- ReX reused из предыдущего слоя (без градиентов)

#### 2.3 Mamba-2 SSD (State Space Model)

```python
# Sequential scan (медленно на Python)
x = mamba_ssm(x)

# Включается через use_mamba=True
# Для seq_len > 512 нужен CUDA kernel
```

### 3. Output Head

```python
# RMSNorm
x = norm(x)

# LM Head (optional BitNet quantization)
logits = lm_head(x)
```

#### 3.1 MTP (Multi-Token Prediction)

Предсказание нескольких токенов вперёд:

```python
for i in range(num_mtp_heads):
    x_i = mtp_norm(x)
    x_i = mtp_proj(x_i)
    mtp_logits[i] = mtp_head(x_i)
```

### 4. BitNet Quantization (b1.58)

```python
# Ternary weights: {-1, 0, +1}
weights = torch.round(torch.clamp(weights, -1, 1))

# STE (Straight-Through Estimator)
# backward pass использует оригинальные веса
```

### 5. CLR (Compressed Latent Reasoning)

Добавляет learnable tokens для рассуждений:

```python
clr_tokens = learnable_tokens(batch_size, num_clr, d_model)
x = torch.cat([clr_tokens, x], dim=1)
```

## Training Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    Training Loop                          │
├────────────────────────────────────────────────────────────┤
│ 1. Data Loading (Streaming)                               │
│    ├─ Infinite loader → batches                             │
│    └─ Tokenizer → input_ids, labels                       │
│                                                            │
│ 2. Forward Pass                                            │
│    ├─ Embedding                                            │
│    ├─ 16 Blocks (Attn + MoE + Mamba)                      │
│    ├─ RMSNorm                                              │
│    └─ LM Head                                              │
│                                                            │
│ 3. Loss Computation                                        │
│    ├─ CrossEntropy (main)                                  │
│    ├─ MTP losses (t+1, t+2)                                │
│    └─ Router z-loss                                        │
│                                                            │
│ 4. Backward Pass (BF16 AMP)                                │
│    ├─ torch.autograd                                       │
│    └─ Gradient checkpointing                               │
│                                                            │
│ 5. Optimizer Step                                          │
│    ├─ AdamW (FP32 state)                                   │
│    └─ Mixed precision (BF16 weights)                      │
│                                                            │
│ 6. Checkpointing                                           │
│    ├─ Save every N steps                                   │
│    └─ Keep top-K checkpoints                               │
└────────────────────────────────────────────────────────────┘
```

## VRAM Optimization

| Technique | VRAM Saved | Impact |
|-----------|------------|--------|
| BF16 AMP | ~50% | None |
| Gradient Checkpointing | ~30% | 10-20% slower |
| use_mamba = False | ~1 GB | Faster training |
| use_bitlinear = True | ~20% | Quality trade-off |
| batch=3 vs batch=5 | ~3 GB | Slower convergence |

### VRAM Monitoring

```python
# Каждый шаг проверяем VRAM
if vram_pct > 88:  # Warn
    torch.cuda.empty_cache()
if vram_pct > 95:  # Critical
    reduce_batch_size()
```

## Checkpoint Format

```python
{
    "step": 1000,
    "config": {...},      # ModelConfig
    "model_state": {...},
    "optimizer_state": {...},
    "metrics": {
        "loss": 2.45,
        "lr": 0.0002,
        "vram_gb": 14.2
    }
}
```

## Data Format

```
data/train/
├── fineweb-000.txt     # Web data (~3GB)
├── c4.txt              # Common Crawl (~1GB)
├── codeparrot.txt      # Code (~600MB)
└── ...
```

Каждый файл — текст, токенизатор автоматически создаётся при первом запуске.

## Inference

```python
# Load model
model = MiniChat(cfg).to(device)
model.load_state_dict(checkpoint["model_state"])

# Generate
input_ids = tokenizer.encode("def factorial(n):")
output = model.generate(input_ids, max_new_tokens=30)
text = tokenizer.decode(output)
```

## Performance

| Config | VRAM | Speed | Tokens/s |
|--------|------|-------|----------|
| batch=5, seq=512 | 14 GB | 0.9 st/s | 60-70 |
| batch=4, seq=512 | 12 GB | 1.0 st/s | 55-65 |
| batch=3, seq=256 | 8 GB | 1.5 st/s | 40-50 |

## Config Options

### Model
- `d_model`: embedding size (768 default)
- `n_layers`: number of layers (16 default)
- `n_heads`: attention heads (12 default)
- `num_experts`: MoE experts (16 default)
- `vocab_size`: vocabulary size (12000 default)

### Training
- `batch_size`: batch size (5 default)
- `seq_len`: sequence length (512 default)
- `learning_rate`: 2e-4 default
- `weight_decay`: 0.1 default

### Features
- `use_mamba`: Enable Mamba SSM
- `use_bitlinear`: BitNet quantization
- `use_rex`: ReX reuse
- `use_mtp`: Multi-token prediction
- `use_gradient_checkpointing`: Memory optimization

## References

- BitNet: https://arxiv.org/abs/2309.05512
- Mamba-2: https://arxiv.org/abs/2405.21060
- DeepSeek-MoE: https://arxiv.org/abs/2401.06066
- RoPE: https://arxiv.org/abs/2104.09864