#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def find_best_checkpoint():
    from bulba1.config import load_config
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    ckpt_dir = PROJECT_ROOT / cfg.checkpoint_dir
    best = ckpt_dir / "best.safetensors"
    if best.exists():
        return str(best)
    files = sorted(
        ckpt_dir.glob("checkpoint_step_*.safetensors"),
        key=lambda f: int(re.search(r"step_(\d+)", f.name).group(1)),
        reverse=True,
    )
    return str(files[0]) if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Checkpoint path (default: best)")
    parser.add_argument("--data", default="data/sft/sft_claude_opus47.jsonl")
    parser.add_argument("--output", default="checkpoints/sft")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    if not ckpt:
        print("No checkpoint found.")
        return
    print(f"Checkpoint: {ckpt}")

    from bulba1.finetune import FineTuner
    ft = FineTuner(ckpt, str(PROJECT_ROOT / "configs" / "default.yaml"))
    ft.sft(args.data, args.output, args.lr, args.epochs, args.batch_size, args.grad_accum)


if __name__ == "__main__":
    main()