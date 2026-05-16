# Configuration Guide

This guide covers all configuration options in Bulba 1, organized by category.

---

## Core Model (Default: 512d, 10 layers)

| Option | Default | Description |
|--------|---------|-------------|
| `d_model` | 512 | Embedding dimension |
| `n_layers` | 10 | Number of transformer layers |
| `n_heads` | 8 | Number of attention heads |
| `vocab_size` | 26000 | Vocabulary size |

---

## Recurrent Architecture (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `num_unique_blocks` | 4 | Unique blocks before repeating |
| `recurrent_repeats` | 3 | How many times to repeat blocks |
| `merge_every_n_layers` | 2 | Token merging frequency |

**Note:** Effective layers = num_unique_blocks × recurrent_repeats

---

## Mixture of Depths (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mixture_of_depths` | true | Enable MoD dynamic gating |
| `mod_capacity` | 0.75 | Token capacity ratio (0-1) |

---

## Token Merging (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `inference_merge_ratio` | 0.3 | Ratio of tokens to merge at inference |
| `merge_every_n_layers` | 2 | Merge frequency |

---

## Mixture of Experts (MoE)

| Option | Default | Description |
|--------|---------|-------------|
| `use_moe` | true | Enable MoE |
| `num_experts` | 4 | Total experts |
| `top_k` | 2 | Experts per token |
| `expert_hidden` | 512 | Expert hidden size |
| `use_rex` | true | Enable ReX (reuse previous layer) |
| `rex_reuse_weight` | 0.1 | ReX blend weight |
| `num_shared_experts` | 3 | Always-active experts |
| `use_grouped_gemm` | false | Use grouped GEMM |

---

## Attention (KDA / DiffAttn)

| Option | Default | Description |
|--------|---------|-------------|
| `use_kda` | true | Use KDA (vs DiffAttn) |
| `use_diff_attn` | true | Enable DiffAttention |
| `use_mla` | true | Multi-Latent Attention |
| `mla_latent_dim` | 32 | MLA latent dimension |
| `use_qk_norm` | true | QK normalization |
| `use_per_head_gating` | true | Per-head gating |

**Position:**
| Option | Default | Description |
|--------|---------|-------------|
| `kda_use_rope` | true | RoPE in KDA |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `max_ctx_len` | 4096 | Max context length |
| `use_sliding_window` | false | Sliding window |
| `sliding_window_size` | 512 | Window size |

**KDA-specific:**
| Option | Default | Description |
|--------|---------|-------------|
| `kda_double_gate` | true | Double gating |
| `kda_use_parallel_scan` | true | Parallel scan (faster) |
| `kda_gate_dim` | 16 | Gate dimension |

---

## Mamba (State Space Model)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mamba` | true | Enable Mamba-3 |
| `mamba_d_state` | 128 | State dimension |
| `mamba_d_conv` | 4 | Convolution width |
| `mamba_expand` | 2 | Expansion factor |
| `attn_every_n_layers` | 4 | Attention frequency |

---

## MHC (Multi-Head Latent Clustering)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mhc` | true | Enable MHC |
| `mhc_n` | 4 | Number of residual streams |
| `mhc_iterations` | 4 | Sinkhorn-Knopp iterations |

---

## MTP (Multi-Token Prediction)

| Option | Default | Description |
|--------|---------|-------------|
| `use_mtp` | true | Enable MTP |
| `num_mtp_heads` | 2 | Number of MTP heads |
| `mtp1_warmup_steps` | 1500 | Head 1 warmup |
| `mtp2_warmup_steps` | 3000 | Head 2 warmup |

---

## Embeddings (NEW v2)

| Option | Default | Description |
|--------|---------|-------------|
| `tied_embeddings` | true | Share LM head with embedding (saves memory) |
| `num_clr_tokens` | 4 | CLR tokens |

---

## Quantization

| Option | Default | Description |
|--------|---------|-------------|
| `use_bitlinear` | true | Enable BitNet quantization |
| `bitnet_activation_bits` | 8 | Activation bits |
| `use_f16` | true | Use FP16/BF16 |
| `use_quantized_kv_cache` | true | Quantize KV cache |
| `kv_cache_bits` | 3 | KV cache bits |

**BitNet a4.8:**
| Option | Default | Description |
|--------|---------|-------------|
| `use_bitnet_a48` | true | Enable 4-bit training |
| `a48_attn_topk_sparsity` | 0.5 | Attention sparsity |
| `a48_use_relu2_glu` | true | Use ReLU² GLU |

---

## Optimizer

| Option | Default | Description |
|--------|---------|-------------|
| `use_muon` | true | Enable Muon optimizer |
| `muon_nesterov` | true | Nesterov momentum |
| `muon_ns_steps` | 3 | Newton-Schulz steps |
| `muon_min_dim` | 256 | Min dimension for Muon |
| `muon_momentum` | 0.95 | Momentum |

**AdamW:**
| Option | Default | Description |
|--------|---------|-------------|
| `learning_rate` | 0.0005 | Learning rate |
| `weight_decay` | 0.1 | Weight decay |
| `beta1` | 0.9 | Adam beta 1 |
| `beta2` | 0.95 | Adam beta 2 |

**EMA:**
| Option | Default | Description |
|--------|---------|-------------|
| `ema_decay` | 0.999 | EMA decay rate |

---

## Training

| Option | Default | Description |
|--------|---------|-------------|
| `seq_len` | 512 | Sequence length |
| `batch_size` | 32 | Batch size |
| `grad_accum_steps` | 2 | Gradient accumulation |
| `total_steps` | 38035 | Total steps |
| `warmup_ratio` | 0.05 | LR warmup ratio |
| `use_lr_cooldown` | false | LR cooldown |
| `use_gradient_checkpointing` | true | Save memory |

**Checkpointing:**
| Option | Default | Description |
|--------|---------|-------------|
| `checkpoint_every` | 1000 | Steps between saves |
| `checkpoint_keep_top_k` | 3 | Keep top K |
| `checkpoint_dir` | "checkpoints/run_bulba1_67m" | Checkpoint path |

---

## Curriculum

| Option | Default | Description |
|--------|---------|-------------|
| `curriculum_warmup_ratio` | 0.15 | Warmup phase |
| `curriculum_start_seq_len` | 128 | Starting sequence length |

---

## Regularization

| Option | Default | Description |
|--------|---------|-------------|
| `dropout` | 0.05 | Dropout |
| `label_smoothing` | 0.05 | Label smoothing |
| `gradient_noise` | 3.0e-5 | Gradient noise |
| `stochastic_depth_prob` | 0.1 | Stochastic depth |
| `token_dropout` | 0.05 | Token dropout |

---

## VRAM Management

| Option | Default | Description |
|--------|---------|-------------|
| `vram_warn_pct` | 88.0 | Warning % |
| `vram_critical_pct` | 95.0 | Critical % |
| `max_batch_reductions` | 6 | Max reductions |

---

## Example Configs

### VRAM Constrained (RTX 3060 12GB)

```yaml
model:
  d_model: 512
  n_layers: 10
  num_experts: 4

training:
  batch_size: 16
  seq_len: 256
  use_gradient_checkpointing: true

model:
  use_mamba: false
  use_bitlinear: true
  tied_embeddings: true
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

### With AutoPilot

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

AutoPilot handles LR scheduling, plateau detection, and warm restarts automatically.

---

## Auto Training Pipeline (NEW v2.1)

After main training completes, automatically run SFT and/or DPO fine-tuning.

| Option | Default | Description |
|--------|---------|-------------|
| `auto_sft` | false | Run SFT after main training |
| `auto_sft_data` | "data/sft" | SFT data directory |
| `auto_sft_epochs` | 3 | SFT epochs |
| `auto_sft_lr` | 1.0e-5 | SFT learning rate |
| `auto_dpo` | false | Run DPO after SFT |
| `auto_dpo_data` | "data/dpo" | DPO data directory |
| `auto_dpo_epochs` | 3 | DPO epochs |
| `auto_dpo_lr` | 1.0e-6 | DPO learning rate |
| `auto_dpo_beta` | 0.1 | DPO beta (KL penalty) |

**Example:**

```yaml
training:
  auto_sft: true
  auto_sft_data: "data/sft"
  auto_sft_epochs: 3
  auto_sft_lr: 1.0e-5

  auto_dpo: true
  auto_dpo_data: "data/dpo"
  auto_dpo_epochs: 3
  auto_dpo_lr: 1.0e-6
  auto_dpo_beta: 0.1
```

**Usage:**

```bash
python -m bulba1.cli --config configs/default.yaml --auto
```

The pipeline runs: **Main Training → SFT → DPO** automatically.