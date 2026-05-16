"""
Bulba1 Configuration Module.

Provides Pydantic-based configuration management with automatic validation
and YAML loading utilities.

Usage:
    from bulba1.config import load_config
    
    cfg = load_config("configs/default.yaml")
    print(cfg.d_model, cfg.learning_rate)
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional
import yaml
from pathlib import Path


class ModelConfig(BaseModel):
    """Main configuration class for Bulba1 model and training."""
    # ── Core architecture ──
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    vocab_size: int = 26000

    # ── MoE ──
    num_experts: int = 4
    top_k: int = 2
    expert_hidden: int = 512
    use_moe: bool = True
    use_rex: bool = True
    rex_reuse_weight: float = 0.1
    num_shared_experts: int = 3
    use_expert_choice: bool = False
    expert_choice_capacity: int = 0

    # ── KDA enhancements ──
    kda_use_rope: bool = True
    kda_double_gate: bool = True

    # ── Attention (DiffAttention) ──
    use_diff_attn: bool = True
    use_mla: bool = True
    mla_latent_dim: int = 32
    use_qk_norm: bool = True
    use_per_head_gating: bool = True
    use_value_residuals: bool = True
    rope_theta: float = 10000.0
    max_ctx_len: int = 4096
    sliding_window_size: int = 512
    lambda_init: float = 0.8

    # ── Mamba ──
    use_mamba: bool = True
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    attn_every_n_layers: int = 4
    use_kda: bool = True
    kda_use_parallel_scan: bool = True
    kda_gate_dim: int = 16

    # ── MHC (DeepSeek) ──
    use_mhc: bool = True
    mhc_n: int = 4
    mhc_iterations: int = 4

    # ── Efficiency techniques ──
    num_unique_blocks: int = 4
    recurrent_repeats: int = 3
    merge_every_n_layers: int = 2
    inference_merge_ratio: float = 0.3
    use_mixture_of_depths: bool = True
    mod_capacity: float = 0.75

    # ── CLR / MTP / Skip-gram ──
    num_clr_tokens: int = 4
    use_mtp: bool = True
    mtp1_warmup_steps: int = 1500
    mtp2_warmup_steps: int = 3000
    num_mtp_heads: int = 2
    use_skip_gram: bool = True
    skip_gram_range: int = 3
    skip_gram_weight: float = 0.05

    # ── Initialization ──
    init_std: float = 0.02
    depth_scaled_init: bool = True

    # ── Loss coefficients ──
    router_z_loss_coef: float = 0.001
    router_entropy_coef: float = 0.001
    attn_z_loss_coef: float = 0.0001
    loss_mtp1_weight: float = 0.3
    loss_mtp2_weight: float = 0.1
    label_smoothing: float = 0.05

    # ── Quantization ──
    use_bitlinear: bool = True
    bitnet_activation_bits: int = 8
    use_f16: bool = True
    use_grouped_gemm: bool = False
    bitlinear_lm_head: bool = False
    bitlinear_mtp: bool = True
    bitnet_init_std: float = 0.001
    use_quantized_kv_cache: bool = True
    kv_cache_bits: int = 3
    use_fp4: bool = False
    use_bitnet_a48: bool = True
    a48_attn_topk_sparsity: float = 0.5
    a48_use_relu2_glu: bool = True
    a48_two_stage_training: bool = False
    a48_stage1_steps_ratio: float = 0.95
    a48_stage1_bits: int = 8
    a48_stage2_bits: int = 4

    # ── Triton Acceleration ──
    use_triton_bitlinear: bool = False

    # ── Optimizer ──
    use_muon: bool = True
    muon_nesterov: bool = True
    muon_ns_steps: int = 3
    muon_min_dim: int = 256
    muon_momentum: float = 0.95
    learning_rate: float = 0.0005
    tied_embeddings: bool = True
    use_mup_init: bool = True
    use_inv_sqrt_lr: bool = True
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    ema_vram_threshold: float = 0.45

    # ── VRAM / OOM safety ──
    vram_warn_pct: float = 88.0
    vram_critical_pct: float = 95.0
    max_batch_reductions: int = 6
    min_batch_size: int = 1
    vram_safety_factor: float = 0.75
    vram_overhead_mb: float = 3072.0
    vram_overhead_factor: float = 1.35

    # ── Training schedule ──
    seq_len: int = 512
    batch_size: int = 32
    skip_preflight: bool = True
    grad_accum_steps: int = 2
    total_steps: int = 38035
    warmup_ratio: float = 0.05
    epochs: int = 3
    use_lr_cooldown: bool = False
    lr_cooldown_ratio: float = 0.05
    use_mtp_cooldown: bool = True
    mtp_cooldown_ratio: float = 0.15
    mtp_end_scale: float = 0.1

    # ── Curriculum ──
    curriculum_warmup_ratio: float = 0.15
    curriculum_start_seq_len: int = 128

    # ── Checkpointing ──
    checkpoint_every: int = 1000
    checkpoint_keep_top_k: int = 3
    checkpoint_dir: str = "checkpoints/run_bulba1_27m"
    log_every: int = 50
    eval_every: int = 500
    eval_max_batches: int = 10
    gen_every: int = 2000

    # ── Regularization ──
    dropout: float = 0.05
    gradient_noise: float = 3e-5
    stochastic_depth_prob: float = 0.1
    token_dropout: float = 0.05

    # ── Data ──
    data_dir: str = "data/tokenized"
    val_data_dir: str = "data/tokenized"
    log_dir: str = "logs"
    num_workers: int = 0
    prefetch_factor: int = 4

    # ── Precision ──
    use_gradient_checkpointing: bool = True
    compile: bool = False

    # ── Chunked CE ──
    chunk_size: int = 8192
    auto_chunked_ce_threshold: int = 0
    ignore_index: int = -100

    # ── Tokenizer IDs ──
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    # ── Auto Training Pipeline ──
    auto_sft: bool = False
    auto_sft_data: str = "data/sft"
    auto_sft_epochs: int = 3
    auto_sft_lr: float = 1e-5
    auto_dpo: bool = False
    auto_dpo_data: str = "data/dpo"
    auto_dpo_epochs: int = 3
    auto_dpo_lr: float = 1e-6
    auto_dpo_beta: float = 0.1

    model_config = ConfigDict(extra="allow")


def load_config(yaml_path: str) -> ModelConfig:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    merged = {}
    merged.update(data.get("model", {}))
    merged.update(data.get("training", {}))
    return ModelConfig(**merged)


def format_params(n: int | None = None) -> str:
    n = n or 0
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000:.1f}K"