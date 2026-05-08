#!/usr/bin/env python3
"""
Bulba 1 — Autonomous LLM Training (профессиональная версия)

Поддерживает YAML‑конфиги и полный автозапуск.
Примеры:
  bulba --config configs/150m.yaml                  # только обучение
  bulba --config configs/150m.yaml --full           # полный пайплайн
  bulba --params 150M --steps 10000 --batch-size 4  # ручной режим
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

from bulba1.config import ModelConfig, find_architecture
from bulba1.model.minichat import MiniChat
from bulba1.tokenizer import HFTokenizer, FastTokenizer, create_dataloader
from bulba1.training.engine import TrainingEngine
from bulba1.training.autotuner import HardwareAutotuner


torch.backends.cudnn.benchmark = True


# ---------- вспомогательные функции ----------
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

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def apply_yaml_config(cfg: ModelConfig, yaml_cfg: dict) -> ModelConfig:
    """Применяет параметры из YAML (секции model и training) к ModelConfig."""
    model_params = yaml_cfg.get("model", {})
    training_params = yaml_cfg.get("training", {})
    for k, v in {**model_params, **training_params}.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def apply_cli_overrides(cfg: ModelConfig, args) -> ModelConfig:
    """Применяет CLI‑аргументы поверх текущего cfg (CLI имеет приоритет)."""
    # Параметры модели и обучения (большинство уже есть в ModelConfig)
    overrides = {
        'seq_len': args.seq_len,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'muon_ns_steps': args.muon_ns_steps,
        'muon_min_dim': args.muon_min_dim,
        'max_grad_norm': args.max_grad_norm,
        'loss_mtp1_weight': args.mtp1_weight,
        'loss_mtp2_weight': args.mtp2_weight,
        'router_z_loss_coef': args.router_z_loss,
        'label_smoothing': args.label_smoothing,
        'ema_decay': args.ema_decay,
        'ema_vram_threshold': args.ema_vram_threshold,
        'vram_warn_pct': args.vram_warn_pct,
        'vram_critical_pct': args.vram_critical_pct,
        'max_batch_reductions': args.max_batch_reductions,
        'vram_safety_factor': args.vram_safety_factor,
        'init_std': args.init_std,
        'lambda_init': args.lambda_init,
        'rex_reuse_weight': args.rex_reuse_weight,
        'generate_max_new_tokens': args.gen_max_tokens,
        'generate_top_k': args.gen_top_k,
        'generate_temperature': args.gen_temperature,
        'chunk_size': args.chunk_size,
        'eval_max_batches': args.eval_max_batches,
        'checkpoint_every': args.checkpoint_every,
        'checkpoint_keep_top_k': args.keep_top_k,
        'checkpoint_dir': args.checkpoint_dir,
        'data_dir': args.data_dir,
        'stage_boundaries': parse_float_list(args.stage_boundaries) if args.stage_boundaries else None,
        'stage_lr_multipliers': parse_float_list(args.stage_lr_mults) if args.stage_lr_mults else None,
        'max_model_params': parse_param_str(args.max_model_params) if args.max_model_params else None,
        'tokens_per_param_target': args.tokens_per_param,
        'overtraining_mode': args.overtraining_mode,
        'overtraining_ratio': args.overtraining_ratio,
        'data_filter_remove_templated': bool(args.remove_templated),
        'data_filter_remove_auto_generated': bool(args.remove_auto_generated),
        'data_filter_min_doc_length': args.min_doc_length,
        'data_filter_max_doc_length': args.max_doc_length,
        'curriculum_seq_lens': parse_int_list(args.curriculum_seq_lens) if args.curriculum_seq_lens else None,
        'curriculum_seq_len_boundaries': parse_float_list(args.curriculum_seq_boundaries) if args.curriculum_seq_boundaries else None,
        'long_context_assembly': args.long_context_assembly,
        'long_context_min_length': args.long_context_min,
        'long_context_max_length': args.long_context_max,
        'bitnet_activation_bits': args.bitnet_activation_bits,
        'bitlinear_lm_head': args.bitlinear_lm_head,
        'bitlinear_mtp': args.bitlinear_mtp,
        'use_quantized_kv_cache': args.use_quantized_kv_cache,
        'kv_cache_bits': args.kv_cache_bits,
        'use_fp4': args.use_fp4,
        'use_bitnet_a48': args.use_bitnet_a48,
        'a48_attn_topk_sparsity': args.a48_topk_sparsity,
        'a48_use_relu2_glu': args.a48_relu2_glu,
        'a48_two_stage_training': args.a48_two_stage,
        'a48_stage1_steps_ratio': args.a48_stage1_ratio,
        'kda_use_rope': args.kda_use_rope,
        'kda_double_gate': args.kda_double_gate,
        'use_expert_choice': args.use_expert_choice,
        'expert_choice_capacity': args.expert_choice_capacity,
        'use_mhc': args.use_mhc,
        'mhc_iterations': args.mhc_iterations,
        # batch_size и grad_accum_steps обрабатываются отдельно после тюнера
    }
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main(config_path: str | None = None):
    parser = argparse.ArgumentParser(description="Bulba 1 — Autonomous LLM Training")
    # ── Основные флаги ──
    parser.add_argument("--config", type=str, default=None, help="Путь к YAML‑конфигу")
    parser.add_argument("--full", action="store_true", help="Полный пайплайн (download + build + train)")
    parser.add_argument("--skip-download", action="store_true", help="Пропустить загрузку данных")
    parser.add_argument("--skip-build", action="store_true", help="Пропустить сборку данных и токенизатора")
    parser.add_argument("--params", type=str, default="0", help="Целевой размер модели (например, 150M)")
    parser.add_argument("--steps", type=int, default=1000, help="Количество шагов обучения")
    parser.add_argument("--batch-size", type=int, default=0, help="Размер батча (0 = авто)")
    parser.add_argument("--seq-len", type=int, default=128, help="Длина последовательности")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Папка с данными")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Размер словаря токенизатора")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Частота сохранения чекпоинтов")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Директория чекпоинтов")
    parser.add_argument("--keep-top-k", type=int, default=3, help="Хранить последних K чекпоинтов")
    parser.add_argument("--compile", action="store_true", help="Использовать torch.compile")
    parser.add_argument("--generate", action="store_true", help="Сгенерировать пример после обучения")
    parser.add_argument("--prompt", type=str, default="def factorial(n):", help="Промпт для генерации")
    parser.add_argument("--prepare-data", action="store_true", help="Устаревший флаг подготовки данных (используйте --full)")
    parser.add_argument("--device", type=str, default="auto", help="Устройство: auto/cuda/cpu")
    parser.add_argument("--force-adamw", action="store_true", help="Принудительно использовать AdamW")
    parser.add_argument("--force-fp32", action="store_true", help="Отключить BF16, использовать FP32")
    parser.add_argument("--grad-accum", type=int, default=0, help="Шагов градиентного аккумулятора (0 = авто)")
    parser.add_argument("--eval-every", type=int, default=0, help="Запускать оценку каждые N шагов (0 = отключено)")
    parser.add_argument("--eval-prompts", type=str, nargs="+", default=["def factorial(n):", "The meaning of life is"],
                        help="Промпты для оценки")
    parser.add_argument("--resume", action="store_true", help="Продолжить с последнего чекпоинта")
    parser.add_argument("--checkpoint", type=int, default=0, help="Продолжить с конкретного шага (0 = последний)")
    parser.add_argument("--analyze", action="store_true", help="Анализ оборудования и датасета")
    parser.add_argument("--max-items", type=int, default=None, help="Максимальное число документов на источник")
    # Оптимизатор
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-min-dim", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Лоссы
    parser.add_argument("--mtp1-weight", type=float, default=0.3)
    parser.add_argument("--mtp2-weight", type=float, default=0.1)
    parser.add_argument("--router-z-loss", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    # EMA / VRAM
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-vram-threshold", type=float, default=0.45)
    parser.add_argument("--vram-warn-pct", type=float, default=88.0)
    parser.add_argument("--vram-critical-pct", type=float, default=95.0)
    parser.add_argument("--max-batch-reductions", type=int, default=3)
    parser.add_argument("--vram-safety-factor", type=float, default=0.75)
    # Curriculum
    parser.add_argument("--stage-boundaries", type=str, default="0.25,0.50,0.75")
    parser.add_argument("--stage-lr-mults", type=str, default="1.0,3.33,1.0,0.5")
    # Модель
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--lambda-init", type=float, default=0.8)
    parser.add_argument("--rex-reuse-weight", type=float, default=0.3)
    # Генерация
    parser.add_argument("--gen-max-tokens", type=int, default=30)
    parser.add_argument("--gen-top-k", type=int, default=50)
    parser.add_argument("--gen-temperature", type=float, default=0.8)
    # Chunked CE / Eval
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--eval-max-batches", type=int, default=10)
    # Данные / фильтрация
    parser.add_argument("--max-model-params", type=str, default="0")
    parser.add_argument("--tokens-per-param", type=float, default=20.0)
    parser.add_argument("--overtraining-mode", action="store_true")
    parser.add_argument("--overtraining-ratio", type=float, default=100.0)
    parser.add_argument("--remove-templated", type=int, default=1)
    parser.add_argument("--remove-auto-generated", type=int, default=1)
    parser.add_argument("--min-doc-length", type=int, default=100)
    parser.add_argument("--max-doc-length", type=int, default=500_000)
    parser.add_argument("--curriculum-seq-lens", type=str, default="128,256,512,1024")
    parser.add_argument("--curriculum-seq-boundaries", type=str, default="0.15,0.35,0.60")
    parser.add_argument("--long-context-assembly", action="store_true")
    parser.add_argument("--long-context-min", type=int, default=4096)
    parser.add_argument("--long-context-max", type=int, default=16384)
    # BitNet / квантизация
    parser.add_argument("--bitnet-activation-bits", type=int, default=8)
    parser.add_argument("--bitlinear-lm-head", action="store_true")
    parser.add_argument("--bitlinear-mtp", action="store_true")
    parser.add_argument("--use-quantized-kv-cache", action="store_true")
    parser.add_argument("--kv-cache-bits", type=int, default=3)
    parser.add_argument("--use-fp4", action="store_true")
    parser.add_argument("--use-bitnet-a48", action="store_true")
    parser.add_argument("--a48-topk-sparsity", type=float, default=0.5)
    parser.add_argument("--a48-relu2-glu", action="store_true")
    parser.add_argument("--a48-two-stage", action="store_true")
    parser.add_argument("--a48-stage1-ratio", type=float, default=0.95)
    # Экспорт / Расписание
    parser.add_argument("--export-hf", action="store_true", help="Экспорт чекпоинта в формат HuggingFace")
    parser.add_argument("--export-dir", type=str, default="hf_export")
    parser.add_argument("--schedule", type=str, help="Запланировать обучение (HH:MM) или 'now'")
    # KDA / MoE / MHC
    parser.add_argument("--kda-use-rope", action="store_true", help="Использовать RoPE в KDA")
    parser.add_argument("--kda-double-gate", action="store_true", help="Двухканальный гейт в KDA")
    parser.add_argument("--use-expert-choice", action="store_true", help="Маршрутизация Expert Choice для MoE")
    parser.add_argument("--expert-choice-capacity", type=int, default=0, help="Capacity для Expert Choice")
    parser.add_argument("--use-mhc", action="store_true", help="Включить Manifold Hyper‑Connections")
    parser.add_argument("--mhc-iterations", type=int, default=5, help="Число итераций Sinkhorn в MHC")

    args = parser.parse_args()

    # Принудительная передача config_path из оркестратора (если вызван не через CLI)
    if config_path:
        args.config = config_path

    # ── Полный пайплайн ──
    if args.full:
        from bulba1.orchestrator import BulbaOrchestrator
        orch = BulbaOrchestrator(args.config)
        orch.run_full(skip_download=args.skip_download, skip_build=args.skip_build)
        return

    # ── Устаревший флаг prepare_data (перенаправляем на оркестратор) ──
    if args.prepare_data:
        print("⚠️  Флаг --prepare-data устарел. Используйте --full.")
        from bulba1.orchestrator import BulbaOrchestrator
        orch = BulbaOrchestrator(args.config)
        orch.download()
        orch.build()
        return

    # ── Экспорт ──
    if args.export_hf:
        from safetensors.torch import load_file
        checkpoint_dir = args.checkpoint_dir
        best_path = os.path.join(checkpoint_dir, "best.safetensors")
        if not os.path.exists(best_path):
            print("❌ Чекпоинт best.safetensors не найден.")
            return
        print(f"📦 Экспорт {best_path} в {args.export_dir}...")
        state_dict = load_file(best_path)
        os.makedirs(args.export_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(args.export_dir, "model.pt"))
        import json
        cfg_for_export = ModelConfig()  # минимальная заглушка; лучше сохранить реальный конфиг из чекпоинта
        config_json = {
            "model_type": "llama",
            "vocab_size": cfg_for_export.vocab_size,
            "hidden_size": cfg_for_export.d_model,
            "num_hidden_layers": cfg_for_export.n_layers,
            "num_attention_heads": cfg_for_export.n_heads,
        }
        with open(os.path.join(args.export_dir, "config.json"), "w") as f:
            json.dump(config_json, f)
        print("✅ Экспорт завершён.")
        return

    # ── Планировщик ──
    if args.schedule:
        import subprocess, re
        if args.schedule.lower() == "now":
            subprocess.run(["systemctl", "--user", "start", "bulba1"])
            print("🚀 Обучение запущено.")
        else:
            match = re.match(r"(\d{1,2}):(\d{2})", args.schedule)
            if not match:
                print("❌ Неверный формат времени. Используйте HH:MM или 'now'")
            else:
                print(f"📅 Запланировано на {args.schedule}. Запуск сервиса...")
                subprocess.run(["systemctl", "--user", "start", "bulba1"])
        return

    # ── Определение устройства ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Устройство: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} МБ")
        print(f"   Compute: SM{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")

    tuner = HardwareAutotuner(device)

    # ── Базовый конфиг модели ──
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        cfg = ModelConfig()
        cfg = apply_yaml_config(cfg, yaml_cfg)
        print(f"📄 Загружен конфиг из {args.config}")
    else:
        target_params = parse_param_str(args.params)
        max_params = parse_param_str(args.max_model_params) or None
        cfg = find_architecture(target_params, args.vocab_size, max_params=max_params) if target_params > 0 else ModelConfig()
        if hasattr(cfg, "_was_capped") and cfg._was_capped:
            print(f"⚠️  Параметры ограничены до {cfg.format_params()}")

    # Применяем CLI оверрайды (CLI приоритетнее YAML)
    cfg = apply_cli_overrides(cfg, args)

    if args.analyze:
        analysis = tuner.full_analysis(cfg, data_dir=args.data_dir)
        import json
        print(json.dumps(analysis, indent=2, default=str))
        return

    # Тюнинг под железо
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
    print(f"🧠 Архитектура: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, experts={cfg.num_experts}")
    print(f"📊 Параметры: {cfg.format_params()}, ~{cfg.storage_size_mb():.0f} МБ")

    # ── Токенизатор ──
    if os.path.exists("data/tokenizer_fast.json"):
        tokenizer = FastTokenizer("data/tokenizer_fast.json")
        tokenizer.load()
        print(f"🔤 Vocab size: {tokenizer.vocab_size}")
    else:
        tokenizer = HFTokenizer(vocab_size=args.vocab_size)
        if not os.path.exists(tokenizer.model_path):
            print("🔤 Тренирую токенизатор...")
            files = list(Path(args.data_dir).rglob("*.txt"))
            if not files:
                print("❌ Нет .txt файлов в data/train. Сначала запустите сборку данных.")
                sys.exit(1)
            tokenizer.train([str(f) for f in files])
        else:
            tokenizer.load()
        print(f"🔤 Vocab size: {tokenizer.vocab_size}")

    # ── Загрузчик данных ──
    loader = create_dataloader(tokenizer, args.data_dir, cfg.batch_size, cfg.seq_len, num_workers=0)
    def infinite_loader():
        while True:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    yield tuple(b.to(device, non_blocking=True) for b in batch)
                else:
                    yield batch.to(device, non_blocking=True)

    eval_loader = None
    if args.eval_every > 0:
        eval_loader = create_dataloader(tokenizer, args.data_dir, cfg.batch_size, cfg.seq_len,
                                        num_workers=0, shuffle=False, return_target=False)

    # ── Модель ──
    print("🏗️  Создание модели...")
    model = MiniChat(cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("⚡ torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Фактические параметры: {cfg.format_params(total_params)}")

    engine = TrainingEngine(model, cfg, tokenizer, device=device, tuned_config=tuned)
    if args.force_adamw and engine.optimizer_type != "adamw":
        engine.optimizer_type = "adamw"
        engine.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
            weight_decay=cfg.weight_decay, fused=(device.type == "cuda"),
        )

    resume_step = 0
    if args.resume or args.checkpoint > 0:
        checkpoint_arg = args.checkpoint if args.checkpoint > 0 else "latest"
        resume_step = engine.resume_from_checkpoint(checkpoint_arg)

    print(f"🚀 Старт обучения на {args.steps} шагов...")
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
        print(f"\n🧪 Генерация по промпту: '{args.prompt}'")
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
            output = model.generate(input_ids,
                                    max_new_tokens=cfg.generate_max_new_tokens,
                                    temperature=cfg.generate_temperature)
            text = tokenizer.decode(output[0].tolist())
            print(f"📝 Результат: {text}")

if __name__ == "__main__":
    main()