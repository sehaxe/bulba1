#!/usr/bin/env python3
"""
Bulba 1 — Autonomous LLM Training (YAML‑only configuration)
"""

import os, sys, argparse, yaml, torch
from pathlib import Path

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.tokenizer import HFTokenizer, FastTokenizer, create_dataloader
from bulba1.training.engine import TrainingEngine

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser(description="Bulba 1 — Autonomous LLM Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, default="def factorial(n):")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-prompts", type=str, nargs="*", default=[])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    # Полный пайплайн (если нужно)
    if args.full:
        from bulba1.orchestrator import BulbaOrchestrator
        BulbaOrchestrator(args.config).run_full(skip_download=args.skip_download, skip_build=args.skip_build)
        return

    # Загружаем YAML
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    all_params = {}
    all_params.update(yaml_cfg.get("model", {}))
    all_params.update(yaml_cfg.get("training", {}))
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

    # Токенизатор
    if os.path.exists("data/tokenizer_fast.json"):
        tokenizer = FastTokenizer("data/tokenizer_fast.json")
        tokenizer.load()
    else:
        tokenizer = HFTokenizer(vocab_size=getattr(cfg, "vocab_size", 26000))
        if not os.path.exists(tokenizer.model_path):
            files = list(Path(cfg.data_dir).rglob("*.txt"))
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
        tokenizer, cfg.data_dir, cfg.batch_size, cfg.seq_len,
        num_workers=num_workers, prefetch_factor=prefetch_factor
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
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    engine = TrainingEngine(model, cfg, tokenizer, device=device)

    resume_step = 0
    if args.resume or args.checkpoint > 0:
        checkpoint_arg = args.checkpoint if args.checkpoint > 0 else "latest"
        resume_step = engine.resume_from_checkpoint(checkpoint_arg)

    print(f"🚀 Старт обучения на {cfg.total_steps} шагов...")
    model = engine.train(
        infinite_loader(),
        eval_loader=None,
        eval_prompts=args.eval_prompts if args.eval_prompts else None,
        resume_step=resume_step,
    )

    if args.generate:
        print(f"\n🧪 Генерация по промпту: '{args.prompt}'")
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
            output = model.generate(input_ids,
                                    max_new_tokens=getattr(cfg, "generate_max_new_tokens", 30),
                                    temperature=getattr(cfg, "generate_temperature", 0.8))
            text = tokenizer.decode(output[0].tolist())
            print(f"📝 Результат: {text}")

if __name__ == "__main__":
    main()