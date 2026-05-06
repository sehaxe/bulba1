from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────────────
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 16
    vocab_size: int = 32000

    # MoE
    num_experts: int = 32
    top_k: int = 2
    expert_hidden: int = 768
    use_moe: bool = True
    use_rex: bool = True
    rex_reuse_weight: float = 0.3
    num_shared_experts: int = 2

    # Attention
    use_diff_attn: bool = True
    use_mla: bool = True
    mla_latent_dim: int = 64
    use_qk_norm: bool = True
    use_per_head_gating: bool = True
    use_value_residuals: bool = True
    rope_theta: float = 10000.0
    max_ctx_len: int = 4096
    use_sliding_window: bool = False
    sliding_window_size: int = 512
    lambda_init: float = 0.8

    # Mamba
    use_mamba: bool = False
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    alternating_pattern: Optional[List[str]] = None
    attn_every_n_layers: int = 4
    use_kda: bool = False
    kda_gate_dim: int = 16

    # MHC
    use_mhc: bool = True
    mhc_iterations: int = 5

    # CLR / MTP / Skip-gram
    num_clr_tokens: int = 4
    use_mtp: bool = True
    num_mtp_heads: int = 2

    use_skip_gram: bool = True
    skip_gram_range: int = 3
    skip_gram_weight: float = 0.05

    # ── Initialization ───────────────────────────────────────────
    init_std: float = 0.02
    depth_scaled_init: bool = False

    # ── Loss coefficients ────────────────────────────────────────
    router_z_loss_coef: float = 0.001
    router_entropy_coef: float = 0.001
    attn_z_loss_coef: float = 0.0001
    loss_mtp1_weight: float = 0.3
    loss_mtp2_weight: float = 0.1
    label_smoothing: float = 0.0

    # ── Curriculum / LR schedule ─────────────────────────────────
    curriculum_warmup_ratio: float = 0.1
    curriculum_start_seq_len: int = 64
    warmup_ratio: float = 0.1
    use_lr_cooldown: bool = False
    lr_cooldown_ratio: float = 0.05

    # Stage boundaries (fraction of total steps) and LR multipliers
    stage_boundaries: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    stage_lr_multipliers: List[float] = field(default_factory=lambda: [1.0, 3.33, 1.0, 0.5])

    # ── Training ─────────────────────────────────────────────────
    seq_len: int = 128
    batch_size: int = 4
    skip_preflight: bool = False  # Skip memory test and use configured batch_size directly
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 1
    grad_accum_steps: int = 0  # 0 = auto from batch_size

    # ── Quantization ─────────────────────────────────────────────
    use_bitlinear: bool = True
    bitnet_activation_bits: int = 8
    use_f16: bool = True
    use_grouped_gemm: bool = False

    bitlinear_lm_head: bool = False
    bitlinear_mtp: bool = False
    bitnet_init_std: float = 0.001
    use_quantized_kv_cache: bool = False
    kv_cache_bits: int = 3
    use_fp4: bool = False
    use_bitnet_a48: bool = False
    a48_attn_topk_sparsity: float = 0.5
    a48_use_relu2_glu: bool = False
    a48_two_stage_training: bool = False
    a48_stage1_steps_ratio: float = 0.95
    a48_stage1_bits: int = 8
    a48_stage2_bits: int = 4

    # ── Optimizer ────────────────────────────────────────────────
    use_muon: bool = True
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_min_dim: int = 256
    muon_momentum: float = 0.95
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    ema_vram_threshold: float = 0.45

    # ── VRAM / OOM safety ────────────────────────────────────────
    vram_warn_pct: float = 88.0
    vram_critical_pct: float = 95.0
    max_batch_reductions: int = 3
    min_batch_size: int = 1
    vram_safety_factor: float = 0.75
    vram_overhead_mb: float = 3072.0
    vram_overhead_factor: float = 1.35

    # ── Checkpointing ────────────────────────────────────────────
    checkpoint_every: int = 50
    checkpoint_keep_top_k: int = 3

    # ── Chunked CE ───────────────────────────────────────────────
    chunk_size: int = 8192
    auto_chunked_ce_threshold: int = 50000
    ignore_index: int = -100

    # ── Generation ───────────────────────────────────────────────
    generate_max_new_tokens: int = 30
    generate_top_k: int = 50
    generate_temperature: float = 0.8

    # ── Data pipeline ────────────────────────────────────────────
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    filtered_dir: str = "data/filtered"
    train_dir: str = "data/train"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_max_docs: int = 1_000_000
    data_max_items_per_source: Optional[int] = None  # None = no limit
    eval_max_batches: int = 10

    # ── Chinchilla / scaling ─────────────────────────────────────
    max_model_params: Optional[int] = None
    tokens_per_param_target: float = 20.0
    overtraining_mode: bool = False
    overtraining_ratio: float = 100.0

    # ── DeepSeek-style data curation ─────────────────────────────
    data_filter_remove_templated: bool = True
    data_filter_min_doc_length: int = 100
    data_filter_max_doc_length: int = 500_000
    data_filter_remove_auto_generated: bool = True
    data_enforce_domain_balance: bool = True
    data_domain_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "code": 0.30,
            "arxiv": 0.20,
            "wiki": 0.15,
            "books": 0.15,
            "web": 0.10,
            "math": 0.10,
        }
    )

    # ── Curriculum / long-context ────────────────────────────────
    curriculum_seq_lens: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    curriculum_seq_len_boundaries: List[float] = field(default_factory=lambda: [0.15, 0.35, 0.60])
    long_context_assembly: bool = True
    long_context_min_length: int = 4096
    long_context_max_length: int = 16384

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer_sample_size: int = 10_000_000
    tokenizer_vocab_candidates: List[int] = field(
        default_factory=lambda: [8000, 12000, 16000, 24000, 32000, 48000, 64000]
    )
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    # ── Methods ──────────────────────────────────────────────────

    def num_params(self) -> int:
        params = self.vocab_size * self.d_model
        attn_params = 4 * self.d_model * self.d_model
        if self.use_mla:
            attn_params += 2 * self.d_model * (self.mla_latent_dim * self.n_heads)
        expert_params = self.num_experts * (
            self.d_model * self.expert_hidden + self.expert_hidden * self.d_model
        )
        shared_expert_params = self.num_shared_experts * (
            2 * self.d_model * self.expert_hidden + self.expert_hidden * self.d_model
        )
        router_params = self.d_model * self.num_experts
        mamba_params = 0
        if self.use_mamba:
            mamba_d_inner = self.d_model * self.mamba_expand
            mamba_params = (
                self.d_model * mamba_d_inner * 3
                + mamba_d_inner * self.mamba_d_state * 2
                + mamba_d_inner * self.mamba_d_conv
            )
        norm_params = self.d_model * 4
        layer_params = (
            attn_params
            + expert_params
            + shared_expert_params
            + router_params
            + mamba_params
            + norm_params
        )
        params += self.n_layers * layer_params
        params += self.d_model * self.vocab_size
        if self.use_mtp:
            params += self.num_mtp_heads * self.d_model * self.vocab_size
            params += max(0, self.num_mtp_heads - 1) * self.d_model * self.d_model
        return params

    def effective_bitness(self) -> float:
        if self.use_bitlinear:
            return 0.95 * 1.58 + 0.045 * 16 + 0.005 * 32
        return 16.0

    def storage_size_mb(self) -> float:
        return self.num_params() * self.effective_bitness() / 8 / 1024 / 1024

    def format_params(self, n: Optional[int] = None) -> str:
        n = n or self.num_params()
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.1f}K"


def find_architecture(
    target_params: int, vocab_size: int = 32000, max_params: Optional[int] = None
) -> ModelConfig:
    effective_target = target_params
    if max_params is not None and max_params > 0:
        effective_target = min(target_params, max_params)
    configs = [
        (512, 8, 8, 8, 512),
        (512, 12, 8, 16, 512),
        (768, 12, 12, 16, 768),
        (768, 16, 16, 32, 768),
        (1024, 16, 16, 32, 1024),
        (1024, 20, 16, 32, 1024),
        (1280, 20, 20, 32, 1280),
    ]
    best_cfg = None
    best_diff = float("inf")
    for d_model, n_layers, n_heads, num_experts, expert_hidden in configs:
        cfg = ModelConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
        )
        diff = abs(cfg.num_params() - effective_target)
        if diff < best_diff:
            best_diff = diff
            best_cfg = cfg
    if best_cfg is not None and max_params is not None and target_params > max_params:
        best_cfg._was_capped = True
    return best_cfg or ModelConfig(vocab_size=vocab_size)
