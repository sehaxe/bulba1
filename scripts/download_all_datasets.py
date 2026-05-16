#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ ЗАГРУЗКА ДАННЫХ ДЛЯ BULBA 150M (datasets + прямые ссылки)
Использует библиотеку datasets для надёжной загрузки BookCorpus, ArXiv,
PhilPapers, MC4, StarCoder, CodeParrot, Claude Opus.
Прямые ссылки оставлены только для GLM‑5.1, Kimi K2.5, MetaMathQA (уже скачаны).
"""

import os, sys, requests, json
import datasets
from huggingface_hub import get_token
from tqdm import tqdm

TMP = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data/tmp"
os.makedirs(TMP, exist_ok=True)

HF_TOKEN = get_token()
if not HF_TOKEN:
    print("❌ Токен не найден. Выполните `hf auth login` или задайте HF_TOKEN.")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------- ЗАГРУЗКА DPO DATASETS ----------
def download_dpo_dataset(name, config, split, max_rows, local_name):
    """Загружает DPO датасет и сохраняет как JSONL с messages форматом."""
    dest = os.path.join(TMP, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print(f"  ⏭️ SKIP {local_name}")
        return True
    print(f"  📥 DPO: {local_name}...")
    try:
        ds = datasets.load_dataset(name, config, split=split, token=HF_TOKEN, trust_remote_code=True)
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=local_name):
                if "messages" in row:
                    f.write(json.dumps({"messages": row["messages"]}) + "\n")
                elif "prompt" in row and "chosen" in row:
                    prompt = row["prompt"]
                    chosen = row["chosen"]
                    if isinstance(chosen, list) and len(chosen) > 0:
                        if isinstance(chosen[0], dict) and "content" in chosen[0]:
                            chosen_text = chosen[0]["content"]
                        else:
                            chosen_text = chosen[0] if chosen else ""
                    else:
                        chosen_text = str(chosen) if chosen else ""
                    msgs = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": chosen_text}]
                    f.write(json.dumps({"messages": msgs}) + "\n")
                elif "conversations" in row:
                    msgs = [{"role": m["from"], "content": m["value"]} for m in row["conversations"]]
                    f.write(json.dumps({"messages": msgs}) + "\n")
        return os.path.getsize(dest) > 100_000
    except Exception as e:
        print(f"  ❌ Ошибка DPO {local_name}: {e}")
        return False


# ---------- ЗАГРУЗКА ЧЕРЕЗ DATASETS ----------
def download_dataset(name, config, split, field, max_rows, local_name):
    """Загружает датасет через `datasets` и сохраняет как JSONL."""
    dest = os.path.join(TMP, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  ⏭️ SKIP {local_name}")
        return True
    print(f"  📥 {local_name} via datasets...")
    try:
        ds = datasets.load_dataset(name, config, split=split, token=HF_TOKEN, trust_remote_code=True)
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        with open(dest, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=local_name):
                text = row.get(field, "")
                if text:
                    f.write(json.dumps({"text": text}) + "\n")
        return os.path.getsize(dest) > 1_000_000
    except Exception as e:
        print(f"  ❌ Ошибка datasets {local_name}: {e}")
        return False

# ---------- ПРЯМАЯ ЗАГРУЗКА (только для проверенных ссылок) ----------
def download_file(url, dest, desc, min_size=1_000_000):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"  ⏭️ SKIP {desc}")
        return True
    print(f"  📥 {desc}")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            if r.status_code == 404:
                print(f"  ❌ 404: {url}")
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        return os.path.getsize(dest) >= min_size
    except Exception as e:
        print(f"  ❌ {desc}: {e}")
        return False

def main():
    print("=" * 60)
    print("📥 ФИНАЛЬНАЯ ЗАГРУЗКА (datasets + прямые ссылки)")
    print("=" * 60)

    # 1. Уже скачанные FineWeb-Edu, C4-en, GLM-5.1, Kimi, MetaMathQA – пропускаем
    # 2. Докачиваем недостающее через datasets

    downloads = [
        # BookCorpus
        ("bookcorpus", "plain_text", "train", "text", 1_000_000, "bookcorpus.jsonl"),
        # ArXiv
        ("scientific_papers", "arxiv", "train", "article", 200_000, "arxiv.jsonl"),
        # PhilPapers
        ("cast42/philarchive", "default", "train", "text", 500_000, "philarchive.jsonl"),
        # MC4 (ru) – 2 шарда прямых ссылок
        ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ru.tfrecord-00059-of-04024.json.gz",
        "mc4_ru_00059.json.gz", "MC4 (ru) part 59", 300_000_000),
        ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ru.tfrecord-00060-of-04024.json.gz",
        "mc4_ru_00060.json.gz", "MC4 (ru) part 60", 300_000_000),
        # MC4 (be) – 1 шард
        ("https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-be.tfrecord-00000-of-00016.json.gz",
        "mc4_be_00000.json.gz", "MC4 (be) part 0", 100_000_000),
        # StarCoder (общий)
        ("bigcode/starcoderdata", "data", "train", "content", 1_000_000, "starcoder.jsonl"),
        # CodeParrot (Python)
        ("codeparrot/github-code", "python", "train", "code", 500_000, "codeparrot.jsonl"),
        # Claude Opus 4.7
        ("TeichAI/lordx64-claude-opus-4.7-max-cleaned", "default", "train", "text", 500_000, "claude_opus47.jsonl"),
    ]

    for name, config, split, field, max_rows, local_name in downloads:
        download_dataset(name, config, split, field, max_rows, local_name)

    # ---------- DPO DATASETS ----------
    print("\n" + "=" * 60)
    print("📥 ЗАГРУЗКА DPO DATASETS")
    print("=" * 60)

    dpo_downloads = [
        # UltraFeedback - best quality preference dataset (20k for small model)
        ("argilla/ultrafeedback-binarized-preferences-cleaned", "default", "train", 20_000, "dpo_ultrafeedback.jsonl"),
        # ORPO-DPO Mix - high quality combined dataset (10k for small model)
        ("mlabonne/orpo-dpo-mix-40k-flat", "default", "train", 10_000, "dpo_orpo_mix.jsonl"),
    ]

    for name, config, split, max_rows, local_name in dpo_downloads:
        download_dpo_dataset(name, config, split, max_rows, local_name)

    # Copy DPO files to data/dpo/
    dpo_dest = "data/dpo"
    os.makedirs(dpo_dest, exist_ok=True)
    for fname in ["dpo_ultrafeedback.jsonl", "dpo_orpo_mix.jsonl"]:
        src = os.path.join(TMP, fname)
        if os.path.exists(src):
            import shutil
            dst = os.path.join(dpo_dest, fname)
            shutil.copy(src, dst)
            print(f"  📦 Скопировано: {fname} -> {dpo_dest}")

    print("\n✅ Загрузка DPO завершена!")
    print("\n✅ Вся загрузка завершена. Запустите сборку: python scripts/build_and_tokenize.py")

if __name__ == "__main__":
    main()