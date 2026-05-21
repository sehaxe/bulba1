"""
Bulba1 CLI - Command Line Interface for training and inference.
"""
import os
# Включаем expandable_segments для борьбы с фрагментацией VRAM
# Это критично для torch.compile + curriculum learning
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message=".*Online softmax.*")
warnings.filterwarnings("ignore", message=".*Not enough SMs.*")

#!/usr/bin/env python3
"""
Bulba1 CLI - Unified command-line interface (ffmpeg-style)
Supports: train, tokenize, chat, profile, monitor, logs, plot, info
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Lazy imports for faster startup
_torch = None
def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def cmd_train(args):
    """Train or resume a model."""
    torch = get_torch()
    from bulba1.config import load_config
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer, HFTokenizer, create_dataloader
    from bulba1.training.engine import TrainingEngine

    torch.backends.cudnn.benchmark = True
    cfg = load_config(args.config)
    print(f"🔧 Loaded config from {args.config}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"💻 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Tokenizer
    data_dir = cfg.data_dir
    if os.path.exists("data/tokenizer_fast.json"):
        tokenizer = FastTokenizer("data/tokenizer_fast.json")
        tokenizer.load()
    else:
        tokenizer = HFTokenizer(vocab_size=cfg.vocab_size)
        if not os.path.exists(tokenizer.model_path):
            files = list(Path(data_dir).rglob("*.txt"))
            if not files:
                sys.exit("❌ No .txt files in data/train")
            tokenizer.train([str(f) for f in files])
        else:
            tokenizer.load()
    print(f"📚 Vocab size: {tokenizer.vocab_size}")

    # DataLoader
    loader = create_dataloader(
        tokenizer, data_dir, cfg.batch_size, cfg.seq_len,
        num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor,
    )

    def infinite_loader():
        while True:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    yield tuple(b.to(device, non_blocking=True) for b in batch)
                else:
                    yield batch.to(device, non_blocking=True)

    # Model
    print("🏗️  Creating model...")
    model = MiniChat(cfg).to(device)
    if cfg.use_f16:
        model = model.to(torch.bfloat16)
    
    if args.compile and hasattr(torch, "compile"):
        # Fix for dynamic shapes (curriculum) and growing lists (attn_res)
        torch._dynamo.config.cache_size_limit = 16
        torch._dynamo.config.accumulated_cache_size_limit = 64
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
        print("⚡ torch.compile enabled")

    # Engine
    engine = TrainingEngine(model, cfg, tokenizer, device=str(device), auto_mode=args.auto)
    
    resume_step = 0
    if args.resume or args.checkpoint > 0:
        checkpoint_arg = args.checkpoint if args.checkpoint > 0 else "latest"
        resume_step = engine.resume_from_checkpoint(checkpoint_arg)
    
    print(f"🚀 Starting training for {cfg.total_steps} steps...")
    try:
        engine.train(infinite_loader(), resume_step=resume_step)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)


def cmd_tokenize(args):
    """Pretokenize and shard datasets."""
    from scripts.pretokenize import tokenize_phase, balance_phase
    from bulba1.tokenizer import FastTokenizer
    
    tokenizer = FastTokenizer(args.tokenizer)
    if not os.path.exists(args.tokenizer):
        print(f"❌ Tokenizer not found: {args.tokenizer}")
        print("   Run training first or specify --tokenizer path")
        sys.exit(1)
    tokenizer.load()
    
    tmp_dir = os.path.join(args.output, '.tmp_domains')
    
    print(f"🚀 Running tokenization pipeline...")
    print(f"   Input: {args.data}")
    print(f"   Output: {args.output}")
    print(f"   Seq len: {args.seq_len}, Stride: {args.stride}")
    
    try:
        tokenize_phase(
            data_dir=args.data,
            tmp_dir=tmp_dir,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            manifest_path=args.manifest,
            log_file=args.log_file,
            domain_weights=None,
        )
        
        balance_phase(
            manifest_path=args.manifest,
            tmp_dir=tmp_dir,
            output_dir=args.output,
            num_shards=args.shards,
            seed=args.seed,
            keep_tmp=False,
        )
        print("✅ Tokenization complete!")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        sys.exit(1)


def cmd_chat(args):
    """Interactive chat / Inference."""
    torch = get_torch()
    from bulba1.config import load_config
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer
    from safetensors.torch import load_file
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    # Load config from checkpoint directory or use default
    config_path = args.config or "configs/default.yaml"
    cfg = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # Load tokenizer
    tokenizer_path = args.tokenizer or "data/tokenizer_fast.json"
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    tokenizer = FastTokenizer(tokenizer_path)
    tokenizer.load()
    
    # Load model
    print(f"🏗️  Loading model from {args.model}...")
    model = MiniChat(cfg).to(device)
    if cfg.use_f16:
        model = model.to(torch.bfloat16)
    
    state_dict = load_file(args.model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✅ Model loaded! Type 'quit' to exit.\n")
    
    # Interactive loop
    while True:
        try:
            prompt = input("👤 You: ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
            
            # Tokenize
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
            
            # Decode
            response = tokenizer.decode(output_ids[0].tolist())
            print(f"🤖 Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def cmd_profile(args):
    """Profile model component latencies."""
    torch = get_torch()
    from bulba1.config import load_config
    from bulba1.model.minichat import MiniChat
    from bulba1.profiler import get_profiler
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔍 Profiling model on {device}...")
    model = MiniChat(cfg).to(device)
    if cfg.use_f16:
        model = model.to(torch.bfloat16)
    
    profiler = get_profiler(device=str(device))
    
    # Dummy input
    x = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=device)
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(3):
        model(x)
    
    # Profile
    print(f"📊 Running {args.steps} iterations...")
    for _ in range(args.steps):
        with profiler.measure("forward_total"):
            model(x)
    
    profiler.report()


def cmd_monitor(args):
    """Start Telegram monitoring bot."""
    token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Bot token required. Use --token or set TELEGRAM_BOT_TOKEN env var")
        sys.exit(1)
    
    try:
        from bulba1.monitor.bot import TrainingMonitorBot
        import asyncio
        
        bot = TrainingMonitorBot(token, args.log_dir, args.chat_id)
        print(f"🤖 Starting Telegram bot...")
        asyncio.run(bot.run())
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: uv add aiogram matplotlib")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped")


def cmd_logs(args):
    """Tail training logs."""
    log_file = Path(args.log_dir) / f"{args.stream}.jsonl"
    if not log_file.exists():
        print(f"⚠️  Log file not found: {log_file}")
        return
    
    print(f"📝 Tailing {log_file} (Ctrl+C to stop)\n")
    
    try:
        with open(log_file, "r") as f:
            f.seek(0, 2)  # Go to end
            while True:
                line = f.readline()
                if line:
                    try:
                        record = json.loads(line)
                        if args.stream == "train":
                            step = record.get('step', '?')
                            total = record.get('total_steps', '?')
                            loss = record.get('loss', 0)
                            tok_s = record.get('tok_per_sec', 0)
                            ts = record.get('time', '?')[-8:]
                            print(f"[{ts}] Step {step}/{total} | loss={loss:.4f} | tok/s={tok_s}")
                        else:
                            print(json.dumps(record, indent=2))
                    except json.JSONDecodeError:
                        print(line.strip())
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n👋 Stopped")


def cmd_plot(args):
    """Generate plots from logs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not installed. Run: uv add matplotlib")
        return
    
    log_file = Path(args.log_dir) / "train.jsonl"
    if not log_file.exists():
        print(f"⚠️  Log file not found: {log_file}")
        return
    
    # Read logs
    steps, losses, ema_losses = [], [], []
    with open(log_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                steps.append(record.get("step", 0))
                losses.append(record.get("loss", 0))
                if record.get("ema_loss"):
                    ema_losses.append(record.get("ema_loss"))
            except json.JSONDecodeError:
                continue
    
    if not steps:
        print("⚠️  No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, losses, alpha=0.4, label="Loss", linewidth=1)
    
    if ema_losses:
        ema_steps = steps[-len(ema_losses):]
        ax.plot(ema_steps, ema_losses, label="EMA Loss", linewidth=2, color="orange")
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training Loss ({args.metric})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output = args.output or f"plots/{args.metric}_plot.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"✅ Plot saved to {output}")


def cmd_info(args):
    """Show model/config information."""
    from bulba1.config import load_config
    
    cfg = load_config(args.config)
    
    print("📊 Bulba1 Configuration")
    print("=" * 60)
    print(f"Model:")
    print(f"  d_model: {cfg.d_model}")
    print(f"  n_layers: {cfg.n_layers}")
    print(f"  n_heads: {cfg.n_heads}")
    print(f"  vocab_size: {cfg.vocab_size}")
    print(f"\nArchitecture:")
    print(f"  MoE: {cfg.use_moe} ({cfg.num_experts} experts, top-{cfg.top_k})")
    print(f"  Mamba: {cfg.use_mamba}")
    print(f"  KDA: {cfg.use_kda}")
    print(f"  AttnRes: {getattr(cfg, 'use_attn_res', False)}")
    print(f"\nTraining:")
    print(f"  batch_size: {cfg.batch_size}")
    print(f"  seq_len: {cfg.seq_len}")
    print(f"  total_steps: {cfg.total_steps}")
    print(f"  learning_rate: {cfg.learning_rate}")
    print(f"\nQuantization:")
    print(f"  BitLinear: {cfg.use_bitlinear}")
    print(f"  Activation bits: {cfg.bitnet_activation_bits}")
    print(f"  F16: {cfg.use_f16}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="bulba",
        description="Bulba1 - Autonomous 1-bit LLM Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- TRAIN ---
    train_p = subparsers.add_parser("train", help="Train or resume a model")
    train_p.add_argument("--config", type=str, default="configs/default.yaml")
    train_p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    train_p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    train_p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    train_p.add_argument("--checkpoint", type=int, default=0, help="Resume from specific step")
    train_p.add_argument("--auto", action="store_true", help="Autopilot mode")
    train_p.set_defaults(func=cmd_train)

    # --- TOKENIZE ---
    tok_p = subparsers.add_parser("tokenize", help="Pretokenize and shard datasets")
    tok_p.add_argument("--data", type=str, default="data/train", help="Input data directory")
    tok_p.add_argument("--output", type=str, default="data/tokenized", help="Output directory")
    tok_p.add_argument("--tokenizer", type=str, default="data/tokenizer_fast.json")
    tok_p.add_argument("--manifest", type=str, default="data_manifest.yaml")
    tok_p.add_argument("--log-file", type=str, default="logs/pretokenize.jsonl")
    tok_p.add_argument("--seq-len", type=int, default=512)
    tok_p.add_argument("--stride", type=int, default=256)
    tok_p.add_argument("--shards", type=int, default=1, help="Number of output shards")
    tok_p.add_argument("--seed", type=int, default=42)
    tok_p.set_defaults(func=cmd_tokenize)

    # --- CHAT ---
    chat_p = subparsers.add_parser("chat", help="Interactive chat / Inference")
    chat_p.add_argument("--model", type=str, required=True, help="Path to safetensors")
    chat_p.add_argument("--config", type=str, help="Config file (default: configs/default.yaml)")
    chat_p.add_argument("--tokenizer", type=str, help="Tokenizer path")
    chat_p.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    chat_p.add_argument("--temperature", type=float, default=0.8)
    chat_p.add_argument("--top-k", type=int, default=50)
    chat_p.add_argument("--thinking", action="store_true", help="Enable Chain of Thought")
    chat_p.add_argument("--speculative", action="store_true", help="Enable speculative decoding")
    chat_p.set_defaults(func=cmd_chat)

    # --- PROFILE ---
    prof_p = subparsers.add_parser("profile", help="Profile model component latencies")
    prof_p.add_argument("--config", type=str, default="configs/default.yaml")
    prof_p.add_argument("--steps", type=int, default=10, help="Number of profiling iterations")
    prof_p.add_argument("--seq-len", type=int, default=128, help="Sequence length for profiling")
    prof_p.set_defaults(func=cmd_profile)

    # --- MONITOR ---
    mon_p = subparsers.add_parser("monitor", help="Start Telegram monitoring bot")
    mon_p.add_argument("--token", type=str, help="Bot token (or set TELEGRAM_BOT_TOKEN env)")
    mon_p.add_argument("--log-dir", type=str, default="logs")
    mon_p.add_argument("--chat-id", type=int, help="Chat ID for updates")
    mon_p.set_defaults(func=cmd_monitor)

    # --- LOGS ---
    logs_p = subparsers.add_parser("logs", help="Tail training logs")
    logs_p.add_argument("--stream", type=str, default="train", choices=["train", "eval", "gen"])
    logs_p.add_argument("--log-dir", type=str, default="logs")
    logs_p.set_defaults(func=cmd_logs)

    # --- PLOT ---
    plot_p = subparsers.add_parser("plot", help="Generate plots from logs")
    plot_p.add_argument("--metric", type=str, default="loss", choices=["loss", "lr", "vram"])
    plot_p.add_argument("--log-dir", type=str, default="logs")
    plot_p.add_argument("--output", type=str, help="Output PNG path")
    plot_p.set_defaults(func=cmd_plot)

    # --- INFO ---
    info_p = subparsers.add_parser("info", help="Show model/config information")
    info_p.add_argument("--config", type=str, default="configs/default.yaml")
    info_p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        if os.environ.get("BULBA_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
