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

    print("\n✅ Загрузка завершена. Запустите сборку: python scripts/build_and_tokenize.py")

if __name__ == "__main__":
    main()