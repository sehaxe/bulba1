#!/usr/bin/env python3
"""
Bulba 1 — Autonomous LLM Training (YAML‑only configuration)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.tokenizer import FastTokenizer, HFTokenizer, create_dataloader
from bulba1.training.engine import TrainingEngine

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Bulba 1 — Autonomous LLM Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-prompts", type=str, nargs="*", default=[])
    parser.add_argument("--auto", action="store_true", help="Автономный режим: AutoLR + плато-детект + warm restart")
    parser.add_argument(
        "--full", action="store_true", help="Запустить полный цикл: загрузка, токенизация, обучение"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Пропустить загрузку датасетов (для --full)"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Пропустить токенизацию (для --full)"
    )
    args = parser.parse_args()

    # Полный пайплайн
    if args.full:
        from bulba1.orchestrator import BulbaOrchestrator

        orch = BulbaOrchestrator(config_path=args.config)
        orch.run_full(skip_download=args.skip_download, skip_build=args.skip_build)
        return

    # Загружаем YAML
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    all_params = {}
    all_params.update(yaml_cfg.get("model", {}))
    all_params.update(yaml_cfg.get("training", {}))
    if args.auto and "autonomy" in yaml_cfg:
        class AutonomyConfig:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        all_params["autonomy"] = AutonomyConfig(**yaml_cfg["autonomy"])
    cfg = ModelConfig(**all_params)
    print(f"📄 Загружен конфиг из {args.config}")

    # Устройство
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Устройство: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    data_dir = getattr(cfg, "data_dir", "data/tokenized")
    batch_size = getattr(cfg, "batch_size", 1)
    seq_len = getattr(cfg, "seq_len", 512)

    # Токенизатор
    if os.path.exists("data/tokenizer_fast.json"):
        tokenizer = FastTokenizer("data/tokenizer_fast.json")
        tokenizer.load()
    else:
        tokenizer = HFTokenizer(vocab_size=getattr(cfg, "vocab_size", 26000))
        if not os.path.exists(tokenizer.model_path):
            files = list(Path(data_dir).rglob("*.txt"))
            if not files:
                sys.exit("❌ Нет .txt файлов в data/train")
            tokenizer.train([str(f) for f in files])
        else:
            tokenizer.load()
    print(f"🔤 Vocab size: {tokenizer.vocab_size}")

    # DataLoader
    num_workers = getattr(cfg, "num_workers", 2)
    prefetch_factor = getattr(cfg, "prefetch_factor", 4)
    loader = create_dataloader(
        tokenizer,
        data_dir,
        batch_size,
        seq_len,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    def infinite_loader():
        while True:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    yield tuple(b.to(device, non_blocking=True) for b in batch)
                else:
                    yield batch.to(device, non_blocking=True)

    # Модель
    print("🏗️  Создание модели...")
    model = MiniChat(cfg).to(device)
    if getattr(cfg, "use_f16", False):
        model = model.to(torch.bfloat16)
        print("🔧 Модель приведена к bfloat16")
    should_compile = args.compile or getattr(cfg, "compile", False)
    if should_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("⚡ torch.compile включён")

    engine = TrainingEngine(model, cfg, tokenizer, device=str(device), auto_mode=args.auto)

    resume_step = 0
    if args.resume or args.checkpoint > 0:
        checkpoint_arg = args.checkpoint if args.checkpoint > 0 else "latest"
        resume_step = engine.resume_from_checkpoint(checkpoint_arg)
    elif os.path.exists(os.path.join(cfg.checkpoint_dir, "checkpoint_step_1000.safetensors")):
        resume_step = engine.resume_from_checkpoint("latest")
        if resume_step > 0:
            print(f"[AUTO-RESUME] Restored from step {resume_step}")

    print(f"🚀 Старт обучения на {cfg.total_steps} шагов...")
    model = engine.train(
        infinite_loader(),
        eval_loader=None,
        eval_prompts=args.eval_prompts if args.eval_prompts else None,
        resume_step=resume_step,
    )

    # Auto SFT after main training
    if getattr(cfg, "auto_sft", False):
        sft_data = getattr(cfg, "auto_sft_data", "data/sft")
        sft_data_file = os.path.join(sft_data, "sft_claude_opus47.jsonl")
        sft_epochs = getattr(cfg, "auto_sft_epochs", 3)
        sft_lr = getattr(cfg, "auto_sft_lr", 1.0e-5)
        
        print(f"\n🎯 Запуск SFT (data={sft_data_file}, epochs={sft_epochs}, lr={sft_lr})...")
        
        import subprocess
        sft_cmd = [
            sys.executable, "scripts/sft_train.py",
            "--data", sft_data_file,
            "--output", "checkpoints/sft",
            "--epochs", str(sft_epochs),
            "--lr", str(sft_lr)
        ]
        subprocess.run(sft_cmd, check=True)
        print(f"✅ SFT завершён!")

    # Auto DPO after SFT
    if getattr(cfg, "auto_dpo", False):
        dpo_data = getattr(cfg, "auto_dpo_data", "data/dpo")
        dpo_data_file = os.path.join(dpo_data, "train.jsonl")
        dpo_epochs = getattr(cfg, "auto_dpo_epochs", 3)
        dpo_lr = getattr(cfg, "auto_dpo_lr", 1.0e-6)
        dpo_beta = getattr(cfg, "auto_dpo_beta", 0.1)
        
        print(f"\n🎯 Запуск DPO (data={dpo_data_file}, epochs={dpo_epochs}, lr={dpo_lr}, beta={dpo_beta})...")
        
        import subprocess
        dpo_cmd = [
            sys.executable, "scripts/dpo_train.py",
            "--data", dpo_data_file,
            "--output", "checkpoints/dpo",
            "--epochs", str(dpo_epochs),
            "--lr", str(dpo_lr),
            "--beta", str(dpo_beta)
        ]
        subprocess.run(dpo_cmd, check=True)
        print(f"✅ DPO завершён!")

    print("\n🏁 Обучение завершено!")



if __name__ == "__main__":
    main()


