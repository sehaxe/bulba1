#!/usr/bin/env python3
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def find_sft_checkpoint():
    for f in ["checkpoints/sft/sft_final.safetensors", "checkpoints/sft/sft_best.safetensors"]:
        p = PROJECT_ROOT / f
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", help="SFT checkpoint path (default: auto-find)")
    parser.add_argument("--data", default="data/sft/sft_claude_opus47.jsonl")
    parser.add_argument("--output", default="checkpoints/dpo")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    sft_ckpt = args.sft_checkpoint or find_sft_checkpoint()
    if not sft_ckpt:
        print("No SFT checkpoint found. Run SFT first.")
        return
    print(f"SFT checkpoint: {sft_ckpt}")

    from bulba1.finetune import FineTuner
    ft = FineTuner(sft_ckpt, str(PROJECT_ROOT / "configs" / "default.yaml"))
    ft.dpo(args.data, args.output, args.lr, args.epochs, args.batch_size, args.grad_accum, args.beta)


if __name__ == "__main__":
    main()