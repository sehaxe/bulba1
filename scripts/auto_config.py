#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def count_tokens(data_dir: str) -> int:
    total = 0
    data_path = Path(data_dir)
    for ext in ("*.bin", "*.txt"):
        for path in data_path.rglob(ext):
            fstat = path.stat()
            if ext == "*.bin":
                total += fstat.st_size // 4
            else:
                total += fstat.st_size // 3
    return total


def estimate_params(
    d_model: int,
    n_layers: int,
    num_experts: int,
    num_shared_experts: int,
    expert_hidden: int,
    vocab_size: int = 26000,
    attn_every: int = 4,
    mamba_d_state: int = 128,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mhc_n: int = 4,
    num_mtp_heads: int = 2,
    num_clr_tokens: int = 4,
    mla_latent_dim: int = 32,
    n_heads: int | None = None,
) -> int:
    if n_heads is None:
        n_heads = max(6, (d_model // 64) & ~1)

    params = 0
    params += vocab_size * d_model * 2
    params += num_clr_tokens * d_model

    num_attn = max(1, n_layers // attn_every) + (1 if n_layers % attn_every > 0 else 0)
    num_mamba = n_layers - num_attn

    mamba_params = (
        d_model * mamba_expand * d_model * 2
        + d_model * mamba_expand * mamba_d_state
        + d_model * d_model * mamba_expand
        + d_model * mamba_expand * mamba_d_conv
        + d_model
    )

    qkv_dim = d_model + 2 * mla_latent_dim
    attn_params = (
        d_model * qkv_dim
        + mla_latent_dim * d_model
        + d_model * d_model
        + n_heads * 2 * d_model
        + d_model * 2
    )

    expert_params = d_model * expert_hidden * 3
    gate_params = d_model * num_experts
    moe_params = gate_params + (num_experts + num_shared_experts) * expert_params

    mhc_params = 2 * d_model * d_model + mhc_n * d_model * d_model

    norm_params = d_model * 2

    for i in range(n_layers):
        params += mhc_params
        if i % attn_every == 0:
            params += attn_params + moe_params + norm_params * 2
        else:
            params += mamba_params + norm_params

    if num_mtp_heads > 0:
        params += num_mtp_heads * d_model * d_model
        params += num_mtp_heads * d_model * vocab_size
        params += d_model

    return params


def estimate_vram(params: int) -> float:
    return params * 220 / (1024**2)


def derive_architecture(
    total_tokens: int,
    vram_mb: float,
    chinchilla_ratio: int = 20,
    attn_every: int = 4,
) -> dict:
    chinchilla_target = total_tokens // chinchilla_ratio
    target_vram_mb = vram_mb * 0.85

    candidates = []
    for d_model in [384, 448, 512, 576, 640, 704, 768]:
        for n_layers in [8, 10, 12, 14, 16]:
            for num_experts in [4, 6, 8]:
                for shared in [1, 2, 3]:
                    for expert_hidden in [d_model, d_model * 2 // 3, d_model * 3 // 2]:
                        if expert_hidden < 128 or expert_hidden > 1024:
                            continue
                        if num_experts <= 2:
                            continue

                        vocab_size = min(max(8000, d_model * 40), 26000)
                        n_heads = max(4, (d_model // 64) & ~1)

                        params = estimate_params(
                            d_model=d_model,
                            n_layers=n_layers,
                            num_experts=num_experts,
                            num_shared_experts=shared,
                            expert_hidden=expert_hidden,
                            vocab_size=vocab_size,
                            attn_every=attn_every,
                            n_heads=n_heads,
                        )

                        vram = estimate_vram(params)

                        if vram > target_vram_mb:
                            continue

                        score = _chinchilla_score(params, chinchilla_target)
                        candidates.append(
                            {
                                "d_model": d_model,
                                "n_layers": n_layers,
                                "n_heads": n_heads,
                                "vocab_size": vocab_size,
                                "num_experts": num_experts,
                                "num_shared_experts": shared,
                                "expert_hidden": expert_hidden,
                                "params": params,
                                "vram_est_mb": vram,
                                "score": score,
                            }
                        )

    if not candidates:
        candidates = [
            {
                "d_model": 384,
                "n_layers": 8,
                "n_heads": 6,
                "vocab_size": 16000,
                "num_experts": 4,
                "num_shared_experts": 2,
                "expert_hidden": 384,
                "params": estimate_params(384, 8, 4, 2, 384, 16000),
                "vram_est_mb": 0,
                "score": 0,
            }
        ]

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def _chinchilla_score(params: int, target: int) -> float:
    if params <= 0 or target <= 0:
        return 0.0
    ratio = params / target
    if ratio > 1.0:
        return 1.0 - (ratio - 1.0)
    return ratio


def write_config(arch: dict, output_path: str, total_tokens: int, vram_mb: float):
    config = {
        "model": {
            "d_model": arch["d_model"],
            "n_layers": arch["n_layers"],
            "n_heads": arch["n_heads"],
            "vocab_size": arch["vocab_size"],
            "num_experts": arch["num_experts"],
            "top_k": 2,
            "expert_hidden": arch["expert_hidden"],
            "use_moe": True,
            "use_rex": True,
            "rex_reuse_weight": 0.1,
            "num_shared_experts": arch["num_shared_experts"],
            "kda_use_rope": True,
            "kda_double_gate": True,
            "use_expert_choice": False,
            "expert_choice_capacity": 0,
            "use_diff_attn": True,
            "use_mla": True,
            "mla_latent_dim": 32,
            "use_qk_norm": True,
            "use_per_head_gating": True,
            "use_value_residuals": True,
            "rope_theta": 10000.0,
            "max_ctx_len": 4096,
            "sliding_window_size": 512,
            "lambda_init": 0.8,
            "use_mamba": True,
            "mamba_d_state": 128,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "attn_every_n_layers": 4,
            "use_kda": True,
            "kda_use_parallel_scan": True,
            "kda_gate_dim": 16,
            "use_mhc": True,
            "mhc_n": 4,
            "mhc_iterations": 4,
            "num_clr_tokens": 4,
            "use_mtp": True,
            "mtp1_warmup_steps": 1500,
            "mtp2_warmup_steps": 3000,
            "num_mtp_heads": 2,
            "use_skip_gram": True,
            "skip_gram_range": 3,
            "skip_gram_weight": 0.05,
            "init_std": 0.02,
            "depth_scaled_init": True,
            "router_z_loss_coef": 0.001,
            "router_entropy_coef": 0.001,
            "attn_z_loss_coef": 0.0001,
            "loss_mtp1_weight": 0.3,
            "loss_mtp2_weight": 0.1,
            "label_smoothing": 0.05,
            "use_bitlinear": True,
            "bitnet_activation_bits": 8,
            "use_f16": True,
            "use_grouped_gemm": False,
            "bitlinear_lm_head": False,
            "bitlinear_mtp": True,
            "bitnet_init_std": 0.001,
            "use_quantized_kv_cache": True,
            "kv_cache_bits": 3,
            "use_fp4": False,
            "use_bitnet_a48": False,
            "a48_attn_topk_sparsity": 0.5,
            "a48_use_relu2_glu": True,
            "a48_two_stage_training": False,
            "a48_stage1_steps_ratio": 0.95,
            "a48_stage1_bits": 8,
            "a48_stage2_bits": 4,
            "use_muon": True,
            "muon_nesterov": True,
            "muon_ns_steps": 3,
            "muon_min_dim": 256,
            "muon_momentum": 0.95,
            "learning_rate": 5e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-8,
            "max_grad_norm": 1.0,
            "ema_decay": 0.999,
            "ema_vram_threshold": 0.45,
            "vram_warn_pct": 88.0,
            "vram_critical_pct": 95.0,
            "max_batch_reductions": 3,
            "min_batch_size": 1,
            "vram_safety_factor": 0.75,
            "vram_overhead_mb": 3072.0,
            "vram_overhead_factor": 1.35,
        },
        "training": {
            "seq_len": 512,
            "batch_size": 32,
            "skip_preflight": False,
            "grad_accum_steps": 2,
            "total_steps": 25000,
            "warmup_ratio": 0.05,
            "use_lr_cooldown": False,
            "lr_cooldown_ratio": 0.05,
            "use_mtp_cooldown": False,
            "mtp_cooldown_ratio": 0.15,
            "mtp_end_scale": 0.1,
            "curriculum_warmup_ratio": 0.02,
            "curriculum_start_seq_len": 256,
            "checkpoint_every": 1000,
            "checkpoint_keep_top_k": 3,
            "checkpoint_dir": "checkpoints/run_auto",
            "log_every": 10,
            "eval_every": 0,
            "eval_max_batches": 10,
            "dropout": 0.05,
            "label_smoothing": 0.05,
            "gradient_noise": 3.0e-5,
            "stochastic_depth_prob": 0.1,
            "data_dir": "data/tokenized",
            "log_dir": "logs",
            "num_workers": 0,
            "prefetch_factor": 4,
            "use_f16": True,
            "use_gradient_checkpointing": True,
            "compile": False,
            "generate_max_new_tokens": 30,
            "generate_top_k": 50,
            "generate_temperature": 0.8,
            "chunk_size": 8192,
            "auto_chunked_ce_threshold": 0,
            "ignore_index": -100,
            "bos_id": 1,
            "eos_id": 2,
            "pad_id": 0,
        },
        "autonomy": {
            "enabled": True,
            "base_lr": 5e-4,
            "plateau_patience": 800,
            "max_lr_reductions": 3,
            "max_warm_restarts": 2,
            "warmup_steps": 1250,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Auto-derived config: {arch['params'] / 1_000_000:.1f}M params, "
                f"~{arch['vram_est_mb']:.0f}MB VRAM\n")
        f.write(f"# Chinchilla optimal: {total_tokens // 20:,} params from {total_tokens:,} tokens\n")
        f.write(f"# VRAM budget: {vram_mb:.0f} MB\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-derive model architecture.")
    parser.add_argument("--data", default="data/tokenized")
    parser.add_argument("--vram", type=float, default=14000)
    parser.add_argument("--output", default="configs/auto_derived.yaml")
    parser.add_argument("--chinchilla-ratio", type=int, default=20)
    args = parser.parse_args()

    total_tokens = count_tokens(args.data)
    if total_tokens < 100_000:
        print(f"WARNING: only {total_tokens} tokens found in {args.data}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Chinchilla optimal: {total_tokens // args.chinchilla_ratio:,} params")
    print(f"VRAM budget: {args.vram} MB (target {args.vram * 0.85:.0f} MB)")

    arch = derive_architecture(total_tokens, args.vram, args.chinchilla_ratio)

    print(f"\nBest fit:")
    print(f"  d_model={arch['d_model']}, n_layers={arch['n_layers']}, "
          f"n_heads={arch['n_heads']}, vocab={arch['vocab_size']}")
    print(f"  experts={arch['num_experts']}+{arch['num_shared_experts']}shared, "
          f"expert_hidden={arch['expert_hidden']}")
    print(f"  ~{arch['params'] / 1_000_000:.1f}M params, "
          f"~{arch['vram_est_mb']:.0f}MB VRAM")

    write_config(arch, args.output, total_tokens, args.vram)


if __name__ == "__main__":
    main()
