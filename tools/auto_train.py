#!/usr/bin/env python3
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bulba1.orchestrator import TrainingOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Bulba 1 — One-command training pipeline")
    parser.add_argument(
        "--data-dir", default="data", help="Data directory (use large drive path for big datasets)"
    )
    parser.add_argument(
        "--model-size", default="766M", help="Model size (e.g., 125M, 766M, 1B, 7B, 13B)"
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Max docs after filtering")
    parser.add_argument(
        "--max-items", type=int, default=None, help="Max items per source (None = no limit)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--analyze", action="store_true", help="Run hardware + dataset analysis before training"
    )
    parser.add_argument("--remove-templated", type=int, default=1)
    parser.add_argument("--remove-auto-generated", type=int, default=1)
    parser.add_argument("--min-doc-length", type=int, default=100)
    parser.add_argument("--max-doc-length", type=int, default=500_000)
    args = parser.parse_args()

    if args.analyze:
        from bulba1.training.autotuner import HardwareAutotuner
        from bulba1.utils.config import ModelConfig
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tuner = HardwareAutotuner(device)
        cfg = ModelConfig()
        analysis = tuner.full_analysis(
            cfg, data_dir=os.path.join(args.data_dir, "train") or args.data_dir
        )

        import json

        print("=" * 60)
        print("HARDWARE + DATASET ANALYSIS")
        print("=" * 60)
        print(json.dumps(analysis, indent=2, default=str))
        return 0

    print("=" * 70)
    print("  BULBA 1 «Singularity» - Automatic Training")
    print("  Launch once. Wait. Get a trained model.")
    print("=" * 70)
    print()

    orchestrator = TrainingOrchestrator(
        model_size=args.model_size,
        total_steps=120_000,
        data_dir=args.data_dir,
        checkpoint_dir=os.path.join(args.data_dir, "checkpoints"),
        log_dir=os.path.join(args.data_dir, "logs"),
        resume=True,
        max_docs=args.max_docs,
        max_items=args.max_items,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        compile_model=not args.no_compile,
        remove_templated=bool(args.remove_templated),
        remove_auto_generated=bool(args.remove_auto_generated),
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
    )

    print("Configuration:")
    print(f"  Model: {args.model_size} parameters")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Max docs: {args.max_docs:,}")
    print(f"  Max items/source: {'unlimited' if args.max_items is None else args.max_items}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Compile: {'Yes' if not args.no_compile else 'No'}")
    print(f"  Training steps: 120,000 (~15B tokens)")
    print(f"  Estimated time: 4-5 days on RTX 5060 Ti")
    print(f"  Resume: Enabled (safe to interrupt and restart)")
    print()
    print("Starting in 3 seconds... Press Ctrl+C to cancel")
    print()

    time.sleep(3)

    success = orchestrator.run()

    if success:
        print(f"\nTraining complete! Check {args.data_dir}/checkpoints/ for your model.")
    else:
        print("\nTraining interrupted. Run again to resume.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
