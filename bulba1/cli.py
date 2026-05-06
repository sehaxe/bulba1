import os
import argparse
from pathlib import Path

import torch

from bulba1.utils.config import ModelConfig, find_architecture
from bulba1.model.minichat import MiniChat
from bulba1.data.tokenizer import HFTokenizer, FastTokenizer, create_dataloader
from bulba1.training.engine import TrainingEngine
from bulba1.training.autotuner import HardwareAutotuner


def parse_param_str(s: str) -> int:
    s = s.strip().upper()
    if s in ("0", "", "AUTO"):
        return 0
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",")]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Bulba 1 — Autonomous LLM Training")
    parser.add_argument(
        "--params", type=str, default="0", help="Target params (e.g., 50M, 1B, 1.5B)"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = auto)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Data directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocab size")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Checkpoint frequency")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--keep-top-k", type=int, default=3, help="Keep top-K checkpoints")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--generate", action="store_true", help="Generate sample after training")
    parser.add_argument("--prompt", type=str, default="def factorial(n):", help="Generation prompt")
    parser.add_argument("--prepare-data", action="store_true", help="Download and prepare datasets")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--force-adamw", action="store_true", help="Force AdamW even if VRAM tight")
    parser.add_argument("--force-fp32", action="store_true", help="Force FP32 (no BF16)")
    parser.add_argument(
        "--grad-accum", type=int, default=0, help="Gradient accumulation steps (0 = auto)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=0, help="Run eval every N steps (0 = disable)"
    )
    parser.add_argument(
        "--eval-prompts",
        type=str,
        nargs="+",
        default=["def factorial(n):", "The meaning of life is"],
        help="Prompts for generation eval",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--analyze", action="store_true", help="Run hardware + dataset analysis and exit"
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="Max dataset items per source (None = no limit)"
    )

    # ── Optimizer ──
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-min-dim", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # ── Loss ──
    parser.add_argument("--mtp1-weight", type=float, default=0.3)
    parser.add_argument("--mtp2-weight", type=float, default=0.1)
    parser.add_argument("--router-z-loss", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # ── EMA ──
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-vram-threshold", type=float, default=0.45)

    # ── OOM / VRAM ──
    parser.add_argument("--vram-warn-pct", type=float, default=88.0)
    parser.add_argument("--vram-critical-pct", type=float, default=95.0)
    parser.add_argument("--max-batch-reductions", type=int, default=3)
    parser.add_argument("--vram-safety-factor", type=float, default=0.75)

    # ── Curriculum ──
    parser.add_argument(
        "--stage-boundaries",
        type=str,
        default="0.25,0.50,0.75",
        help="Stage boundaries (comma-separated)",
    )
    parser.add_argument(
        "--stage-lr-mults", type=str, default="1.0,3.33,1.0,0.5", help="Stage LR multipliers"
    )

    # ── Model ──
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--lambda-init", type=float, default=0.8)
    parser.add_argument("--rex-reuse-weight", type=float, default=0.3)

    # ── Generation ──
    parser.add_argument("--gen-max-tokens", type=int, default=30)
    parser.add_argument("--gen-top-k", type=int, default=50)
    parser.add_argument("--gen-temperature", type=float, default=0.8)

    # ── Chunked CE ──
    parser.add_argument("--chunk-size", type=int, default=8192)

    # ── Eval ──
    parser.add_argument("--eval-max-batches", type=int, default=10)

    parser.add_argument(
        "--max-model-params", type=str, default="0", help="Hard cap on model params (0 = no cap)"
    )
    parser.add_argument(
        "--tokens-per-param",
        type=float,
        default=20.0,
        help="Target tokens per parameter (Chinchilla)",
    )
    parser.add_argument(
        "--overtraining-mode", action="store_true", help="Enable overtraining when data-rich"
    )
    parser.add_argument(
        "--overtraining-ratio", type=float, default=100.0, help="Tokens/param for overtraining"
    )
    parser.add_argument(
        "--remove-templated", type=int, default=1, help="Remove templated content (0/1)"
    )
    parser.add_argument(
        "--remove-auto-generated", type=int, default=1, help="Remove AI-generated markers (0/1)"
    )
    parser.add_argument("--min-doc-length", type=int, default=100)
    parser.add_argument("--max-doc-length", type=int, default=500_000)
    parser.add_argument(
        "--curriculum-seq-lens", type=str, default="128,256,512,1024", help="Seq lens per stage"
    )
    parser.add_argument(
        "--curriculum-seq-boundaries", type=str, default="0.15,0.35,0.60", help="Stage boundaries"
    )
    parser.add_argument("--long-context-assembly", action="store_true", help="Assemble long docs")
    parser.add_argument("--long-context-min", type=int, default=4096)
    parser.add_argument("--long-context-max", type=int, default=16384)
    parser.add_argument(
        "--bitnet-activation-bits",
        type=int,
        default=8,
        help="BitNet activation quantization bits (8=INT8, 4=INT4)",
    )
    parser.add_argument(
        "--bitlinear-lm-head",
        action="store_true",
        help="Quantize lm_head weights (default: off for quality)",
    )
    parser.add_argument(
        "--bitlinear-mtp", action="store_true", help="Quantize MTP projection/head weights"
    )
    parser.add_argument(
        "--use-quantized-kv-cache", action="store_true", help="Use 3-bit quantized KV cache"
    )
    parser.add_argument(
        "--kv-cache-bits", type=int, default=3, help="KV cache quantization bits (3 or 4)"
    )
    parser.add_argument(
        "--use-fp4", action="store_true", help="Use FP4 quantization for activations"
    )
    parser.add_argument("--use-bitnet-a48", action="store_true", help="Enable BitNet a4.8 features")
    parser.add_argument(
        "--a48-topk-sparsity", type=float, default=0.5, help="Attention Top-K sparsity ratio"
    )
    parser.add_argument("--a48-relu2-glu", action="store_true", help="Use ReLU^2-GLU in FFN")
    parser.add_argument(
        "--a48-two-stage", action="store_true", help="Two-stage training: 8-bit -> 4-bit"
    )
    parser.add_argument("--a48-stage1-ratio", type=float, default=0.95, help="Stage 1 step ratio")

    parser.add_argument(
        "--export-hf", action="store_true", help="Export checkpoint to HuggingFace format"
    )
    parser.add_argument("--export-dir", type=str, default="hf_export", help="Export directory")
    parser.add_argument("--schedule", type=str, help="Schedule training at time (HH:MM) or 'now'")

    args = parser.parse_args()

    if args.prepare_data:
        from scripts.prepare_data import prepare_datasets

        prepare_datasets(args.data_dir, args.max_items)
        return

    if args.schedule:
        import subprocess
        import datetime

        if args.schedule.lower() == "now":
            subprocess.run(["systemctl", "--user", "start", "bulba1"])
            print("Started training")
        else:
            import re

            match = re.match(r"(\d{1,2}):(\d{2})", args.schedule)
            if not match:
                print("Invalid time format. Use HH:MM")
                return
            print("Scheduling training...")
            subprocess.run(["systemctl", "--user", "start", "bulba1"])
            print("Training started")
        return

    if args.export_hf:
        import subprocess
        from safetensors.torch import load_file

        checkpoint_dir = args.checkpoint_dir
        best_path = f"{checkpoint_dir}/best.safetensors"
        if not os.path.exists(best_path):
            print("No checkpoint found")
            return
        print(f"Exporting {best_path} to {args.export_dir}...")
        # Load and resave as PyTorch format
        state_dict = load_file(best_path)
        os.makedirs(args.export_dir, exist_ok=True)
        torch.save(state_dict, f"{args.export_dir}/model.pt")
        # Create config.json
        config = {
            "model_type": "llama",
            "vocab_size": cfg.vocab_size if hasattr(cfg, "vocab_size") else 32000,
            "hidden_size": cfg.d_model if hasattr(cfg, "d_model") else 768,
            "num_hidden_layers": cfg.n_layers if hasattr(cfg, "n_layers") else 16,
            "num_attention_heads": cfg.n_heads if hasattr(cfg, "n_heads") else 12,
        }
        import json

        with open(f"{args.export_dir}/config.json", "w") as f:
            json.dump(config, f)
        print(f"Exported to {args.export_dir}/")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
        print(
            f"Compute: SM{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        )

    tuner = HardwareAutotuner(device)

    target_params = parse_param_str(args.params)
    max_params = parse_param_str(args.max_model_params)
    if max_params == 0:
        max_params = None
    cfg = (
        find_architecture(target_params, args.vocab_size, max_params=max_params)
        if target_params > 0
        else ModelConfig()
    )
    if hasattr(cfg, "_was_capped") and cfg._was_capped:
        print(
            f"WARNING: Target params {args.params} exceeds max cap {args.max_model_params}. Using {cfg.format_params()} instead."
        )

    # Apply all CLI overrides to config
    cfg.seq_len = args.seq_len
    cfg.learning_rate = args.lr
    cfg.checkpoint_every_n_layers = 1
    cfg.weight_decay = args.weight_decay
    cfg.beta1 = args.beta1
    cfg.beta2 = args.beta2
    cfg.muon_ns_steps = args.muon_ns_steps
    cfg.muon_min_dim = args.muon_min_dim
    cfg.max_grad_norm = args.max_grad_norm
    cfg.loss_mtp1_weight = args.mtp1_weight
    cfg.loss_mtp2_weight = args.mtp2_weight
    cfg.router_z_loss_coef = args.router_z_loss
    cfg.label_smoothing = args.label_smoothing
    cfg.ema_decay = args.ema_decay
    cfg.ema_vram_threshold = args.ema_vram_threshold
    cfg.vram_warn_pct = args.vram_warn_pct
    cfg.vram_critical_pct = args.vram_critical_pct
    cfg.max_batch_reductions = args.max_batch_reductions
    cfg.vram_safety_factor = args.vram_safety_factor
    cfg.init_std = args.init_std
    cfg.lambda_init = args.lambda_init
    cfg.rex_reuse_weight = args.rex_reuse_weight
    cfg.generate_max_new_tokens = args.gen_max_tokens
    cfg.generate_top_k = args.gen_top_k
    cfg.generate_temperature = args.gen_temperature
    cfg.chunk_size = args.chunk_size
    cfg.eval_max_batches = args.eval_max_batches
    cfg.checkpoint_every = args.checkpoint_every
    cfg.checkpoint_keep_top_k = args.keep_top_k
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.data_dir = args.data_dir
    cfg.stage_boundaries = parse_float_list(args.stage_boundaries)
    cfg.stage_lr_multipliers = parse_float_list(args.stage_lr_mults)
    cfg.max_model_params = parse_param_str(args.max_model_params)
    cfg.tokens_per_param_target = args.tokens_per_param
    cfg.overtraining_mode = args.overtraining_mode
    cfg.overtraining_ratio = args.overtraining_ratio
    cfg.data_filter_remove_templated = bool(args.remove_templated)
    cfg.data_filter_remove_auto_generated = bool(args.remove_auto_generated)
    cfg.data_filter_min_doc_length = args.min_doc_length
    cfg.data_filter_max_doc_length = args.max_doc_length
    cfg.curriculum_seq_lens = parse_int_list(args.curriculum_seq_lens)
    cfg.curriculum_seq_len_boundaries = parse_float_list(args.curriculum_seq_boundaries)
    cfg.long_context_assembly = args.long_context_assembly
    cfg.long_context_min_length = args.long_context_min
    cfg.long_context_max_length = args.long_context_max
    cfg.bitnet_activation_bits = args.bitnet_activation_bits
    cfg.bitlinear_lm_head = args.bitlinear_lm_head
    cfg.bitlinear_mtp = args.bitlinear_mtp
    cfg.use_quantized_kv_cache = args.use_quantized_kv_cache
    cfg.kv_cache_bits = args.kv_cache_bits
    cfg.use_fp4 = args.use_fp4
    cfg.use_bitnet_a48 = args.use_bitnet_a48
    cfg.a48_attn_topk_sparsity = args.a48_topk_sparsity
    cfg.a48_use_relu2_glu = args.a48_relu2_glu
    cfg.a48_two_stage_training = args.a48_two_stage
    cfg.a48_stage1_steps_ratio = args.a48_stage1_ratio

    if args.analyze:
        analysis = tuner.full_analysis(cfg, data_dir=args.data_dir)
        import json

        print(json.dumps(analysis, indent=2, default=str))
        return

    tuned = tuner.autotune(cfg)
    if args.batch_size > 0:
        tuned.batch_size = args.batch_size
    if args.force_adamw:
        tuned.optimizer_type = "adamw"
    if args.force_fp32:
        tuned.use_bf16 = False

    cfg.batch_size = tuned.batch_size
    cfg.use_f16 = tuned.use_bf16
    cfg.use_gradient_checkpointing = tuned.use_gradient_checkpointing
    cfg.use_muon = tuned.optimizer_type == "muon"
    cfg.grad_accum_steps = args.grad_accum if args.grad_accum > 0 else max(1, tuned.batch_size)

    print(f"\n{tuner.report(tuned)}\n")

    print(
        f"Architecture: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, experts={cfg.num_experts}"
    )
    print(f"Parameters: {cfg.format_params()}")
    print(f"Storage: ~{cfg.storage_size_mb():.0f} MB")

    if os.path.exists("data/tokenizer_fast.json"):
        tokenizer = FastTokenizer("data/tokenizer_fast.json")
        tokenizer.load()
        print(f"Vocab size: {tokenizer.vocab_size}")
    else:
        tokenizer = HFTokenizer(vocab_size=args.vocab_size)
        if not os.path.exists(tokenizer.model_path):
            print("Training tokenizer...")
            files = list(Path(args.data_dir).rglob("*.txt"))
            if not files:
                print(f"No .txt files found in {args.data_dir}. Run with --prepare-data first.")
                os.makedirs(args.data_dir, exist_ok=True)
                with open(os.path.join(args.data_dir, "dummy.txt"), "w") as f:
                    f.write("hello world\n" * 1000)
                files = [os.path.join(args.data_dir, "dummy.txt")]
            tokenizer.train([str(f) for f in files])
        else:
            tokenizer.load()
        print(f"Vocab size: {tokenizer.vocab_size}")

    loader = create_dataloader(tokenizer, args.data_dir, cfg.batch_size, cfg.seq_len, num_workers=0)

    print("Building model...")
    model = MiniChat(cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Actual parameters: {cfg.format_params(total_params)}")

    def infinite_loader():
        while True:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    yield tuple(b.to(device, non_blocking=True) for b in batch)
                else:
                    yield batch.to(device, non_blocking=True)

    eval_loader = None
    if args.eval_every > 0:
        eval_loader = create_dataloader(
            tokenizer,
            args.data_dir,
            cfg.batch_size,
            cfg.seq_len,
            num_workers=0,
            shuffle=False,
            return_target=False,
        )

    engine = TrainingEngine(model, cfg, tokenizer, device=device, tuned_config=tuned)
    if args.force_adamw and engine.optimizer_type != "adamw":
        engine.optimizer_type = "adamw"
        engine.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            fused=(device.type == "cuda"),
        )

    resume_step = 0
    if args.resume:
        resume_step = engine.resume_from_checkpoint("latest")

    print(f"Starting training for {args.steps} steps...")
    model = engine.train(
        infinite_loader(),
        args.steps,
        eval_loader=eval_loader,
        eval_every=args.eval_every,
        eval_prompts=args.eval_prompts,
        resume_step=resume_step,
        checkpoint_every=args.checkpoint_every,
    )

    if args.generate:
        print(f"\nGenerating from prompt: '{args.prompt}'")
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(
                [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
            )
            output = model.generate(
                input_ids,
                max_new_tokens=cfg.generate_max_new_tokens,
                temperature=cfg.generate_temperature,
            )
            text = tokenizer.decode(output[0].tolist())
            print(f"Generated: {text}")


if __name__ == "__main__":
    main()
