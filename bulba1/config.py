"""
Bulba1 Configuration with auto-architecture generation.
"""
import math, warnings
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_CONFIG_CACHE = {}

def _find_beautiful_params(target):
    configs = []
    for d_model in [384, 512, 640, 768, 896, 1024, 1280, 1536, 2048]:
        for n_layers in [8, 10, 12, 16, 20, 24, 32]:
            for n_heads in [6, 8, 12, 16, 20, 24, 32]:
                if d_model % n_heads != 0: continue
                if d_model // n_heads not in [32, 64, 128]: continue
                base = 12 * (d_model ** 2) * n_layers
                if target > 100_000_000:
                    num_experts = 8 if target < 500_000_000 else (16 if target < 2_000_000_000 else 32)
                    params = int(base * (num_experts * 0.5))
                else:
                    params = base
                vocab = 32000 if target > 100_000_000 else 26000
                params += vocab * d_model * 2
                configs.append({"params": params, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "num_experts": num_experts if target > 100_000_000 else 1})
    return min(configs, key=lambda x: abs(x["params"] - target))

def _format_params(n):
    if n >= 1_000_000_000: return f"{n/1e9:.1f}B"
    elif n >= 1_000_000: return f"{n/1e6:.1f}M"
    return f"{n/1e3:.1f}K"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    
    target_params: str | None = None
    auto_architecture: bool = True
    d_model: int = Field(512, gt=0)
    n_layers: int = Field(10, gt=0)
    n_heads: int = Field(8, gt=0)
    vocab_size: int = Field(26000, gt=0)
    num_experts: int = Field(1, ge=1)
    top_k: int = Field(2, ge=1)
    expert_hidden: int = Field(512, gt=0)
    use_moe: bool = True
    num_shared_experts: int = Field(2, ge=0)
    use_diff_attn: bool = True
    use_mla: bool = True
    mla_latent_dim: int = Field(64, gt=0)
    use_qk_norm: bool = True
    rope_theta: float = 10000.0
    max_ctx_len: int = Field(32768, gt=0)
    use_kimi_linear: bool = True
    kimi_local_window: int = 512
    kimi_gate_dim: int = 16
    kimi_double_gate: bool = True
    use_attn_res: bool = True
    attn_res_mode: Literal["auto", "full", "block"] = "auto"
    attn_res_num_blocks: int = Field(8, gt=0)
    attn_res_recency_bias_init: float = 10.0
    use_mtp: bool = True
    num_mtp_heads: int = Field(4, ge=1, le=8)
    use_bitlinear: bool = True
    use_bitnet_a48: bool = True
    bitnet_activation_bits: int = 8
    use_muon: bool = True
    learning_rate: float = 0.001
    seq_len: int = 2048
    batch_size: int = 32
    total_steps: int = 50000
    curriculum_warmup_ratio: float = 0.1
    curriculum_start_seq_len: int = 128
    
    # Additional fields from original config
    use_rex: bool = False
    rex_reuse_weight: float = 0.1
    kda_use_rope: bool = False
    kda_double_gate: bool = False
    use_expert_choice: bool = False
    expert_choice_capacity: int = 0
    use_per_head_gating: bool = True
    use_value_residuals: bool = True
    sliding_window_size: int = 512
    lambda_init: float = 0.8
    use_mamba: bool = False
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    attn_every_n_layers: int = 4
    use_kda: bool = False
    kda_use_parallel_scan: bool = False
    kda_gate_dim: int = 16
    num_unique_blocks: int = 4
    recurrent_repeats: int = 1
    merge_every_n_layers: int = 2
    inference_merge_ratio: float = 0.3
    use_mixture_of_depths: bool = False
    mod_capacity: float = 0.75
    num_clr_tokens: int = 0
    mtp1_warmup_steps: int = 1000
    mtp2_warmup_steps: int = 3000
    mtp3_warmup_steps: int = 6000
    mtp4_warmup_steps: int = 10000
    loss_mtp1_weight: float = 0.4
    loss_mtp2_weight: float = 0.3
    loss_mtp3_weight: float = 0.2
    loss_mtp4_weight: float = 0.1
    use_skip_gram: bool = False
    skip_gram_range: int = 3
    skip_gram_weight: float = 0.05
    init_std: float = 0.02
    depth_scaled_init: bool = True
    use_mup_init: bool = True
    router_z_loss_coef: float = 0.001
    router_entropy_coef: float = 0.001
    attn_z_loss_coef: float = 0.0001
    label_smoothing: float = 0.05
    use_f16: bool = True
    use_grouped_gemm: bool = False
    bitlinear_lm_head: bool = False
    bitlinear_mtp: bool = True
    bitnet_init_std: float = 0.001
    use_quantized_kv_cache: bool = True
    kv_cache_bits: int = 3
    use_fp4: bool = False
    a48_attn_topk_sparsity: float = 0.5
    a48_use_relu2_glu: bool = True
    a48_two_stage_training: bool = True
    a48_stage1_steps_ratio: float = 0.9
    a48_stage1_bits: int = 8
    a48_stage2_bits: int = 4
    use_triton_bitlinear: bool = False
    distributed: bool = False
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "bulba1"
    muon_nesterov: bool = True
    muon_ns_steps: int = 5  # Paper 2502.16982 §2.2
    muon_min_dim: int = 2    # Paper 2502.16982 Alg 1
    muon_momentum: float = 0.95
    tied_embeddings: bool = True
    use_inv_sqrt_lr: bool = True
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    ema_vram_threshold: float = 0.45
    vram_warn_pct: float = 88.0
    vram_critical_pct: float = 95.0
    max_batch_reductions: int = 6
    min_batch_size: int = 1
    vram_safety_factor: float = 0.75
    vram_overhead_mb: float = 3072.0
    vram_overhead_factor: float = 1.35
    skip_preflight: bool = True
    grad_accum_steps: int = 2
    warmup_ratio: float = 0.05
    epochs: int = 3
    use_lr_cooldown: bool = False
    lr_cooldown_ratio: float = 0.05
    use_mtp_cooldown: bool = True
    mtp_cooldown_ratio: float = 0.15
    mtp_end_scale: float = 0.1
    checkpoint_every: int = 1000
    checkpoint_keep_top_k: int = 3
    checkpoint_dir: str = "checkpoints/run_bulba1_ultimate"
    log_every: int = 50
    eval_every: int = 500
    eval_max_batches: int = 10
    gen_every: int = 2000
    dropout: float = 0.05
    gradient_noise: float = 3e-5
    stochastic_depth_prob: float = 0.1
    token_dropout: float = 0.05
    data_dir: str = "data/tokenized"
    val_data_dir: str = "data/tokenized"
    log_dir: str = "logs"
    num_workers: int = 0
    prefetch_factor: int = 4
    use_gradient_checkpointing: bool = True
    compile: bool = False
    chunk_size: int = 8192
    auto_chunked_ce_threshold: int = 0
    ignore_index: int = -100
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0
    auto_sft: bool = False
    auto_sft_data: str = "data/sft"
    auto_sft_epochs: int = 3
    auto_sft_lr: float = 1e-5
    auto_dpo: bool = False
    auto_dpo_data: str = "data/dpo"
    auto_dpo_epochs: int = 3
    auto_dpo_lr: float = 1e-6
    auto_dpo_beta: float = 0.1
    
    @model_validator(mode="after")
    def auto_configure(self):
        if self.target_params and self.auto_architecture:
            target = self._parse_params(self.target_params)
            arch = _find_beautiful_params(target)
            self.d_model = arch["d_model"]
            self.n_layers = arch["n_layers"]
            self.n_heads = arch["n_heads"]
            if arch["num_experts"] > 1:
                self.num_experts = arch["num_experts"]
                self.use_moe = True
            else:
                self.num_experts = 1
                self.use_moe = False
            self.vocab_size = 32000 if target > 100_000_000 else 26000
            self.use_mla = self.d_model >= 1024
            if self.attn_res_mode == "auto":
                self.attn_res_mode = "full" if self.n_layers <= 12 else "block"
            print(f"🎯 Auto-generated: {_format_params(arch['params'])} params")
            print(f"   d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}")
            print(f"   MoE: {self.num_experts} experts, AttnRes: {self.attn_res_mode}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        return self
    
    def _parse_params(self, s):
        s = s.lower().strip()
        if s.endswith("b"): return int(float(s[:-1]) * 1_000_000_000)
        elif s.endswith("m"): return int(float(s[:-1]) * 1_000_000)
        elif s.endswith("k"): return int(float(s[:-1]) * 1_000)
        return int(s)
    
    def format_params(self):
        n = 12 * (self.d_model ** 2) * self.n_layers
        if self.use_moe and self.num_experts > 1:
            n = int(n * (self.num_experts * 0.5))
        return _format_params(n)


def load_config(yaml_path):
    path = Path(yaml_path)
    cache_key = str(path.resolve())
    if cache_key in _CONFIG_CACHE: return _CONFIG_CACHE[cache_key]
    if not path.exists(): raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = {}
    if isinstance(data, dict):
        merged.update(data.get("model", {}) or {})
        merged.update(data.get("training", {}) or {})
    cfg = ModelConfig(**merged)
    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def generate_config(target_params, output_path=None):
    cfg = ModelConfig(target_params=target_params, auto_architecture=True)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg.model_dump(), f, default_flow_style=False)
        print(f"✅ Config saved to {output_path}")
    return cfg


def clear_config_cache(): _CONFIG_CACHE.clear()
