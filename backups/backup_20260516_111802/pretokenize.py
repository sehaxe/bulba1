#!/usr/bin/env python3
"""
МАКСИМАЛЬНО НАДЁЖНАЯ ПРЕДТОКЕНИЗАЦИЯ — не падает на больших файлах.
Обрабатывает каждый txt последовательно, батчами по 1000 строк,
сразу сбрасывает токены на диск. Памяти хватает с огромным запасом.
"""

import os, sys, glob
import numpy as np
from tqdm import tqdm
from bulba1.tokenizer import FastTokenizer

DATA_TRAIN = "data/train"
TOKENIZER_PATH = "data/tokenizer_fast.json"
BIN_OUT = "data/tokenized"
CHUNK_LINES = 1000          # очень маленький батч – память < 50 МБ

os.makedirs(BIN_OUT, exist_ok=True)

tok = FastTokenizer(TOKENIZER_PATH)
tok.load()

def process_file(txt_path: str) -> str:
    bin_name = os.path.basename(txt_path).replace(".txt", ".bin")
    bin_path = os.path.join(BIN_OUT, bin_name)
    if os.path.exists(bin_path):
        return f"⏭️ SKIP {bin_name} (уже готов)"

    print(f"🔤 Обрабатываю {bin_name} …")
    total_tokens = 0

    with open(txt_path, encoding="utf-8") as fin, \
         open(bin_path, "wb") as fout:

        batch = []
        for line in tqdm(fin, desc=bin_name, unit=" строк"):
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= CHUNK_LINES:
                # Токенизируем и сразу пишем на диск
                encoded = tok.encode_batch(batch)
                for ids in encoded:
                    if ids:
                        arr = np.array(ids, dtype=np.int32)
                        arr.tofile(fout)
                        total_tokens += len(ids)
                batch = []

        # Остатки
        if batch:
            encoded = tok.encode_batch(batch)
            for ids in encoded:
                if ids:
                    arr = np.array(ids, dtype=np.int32)
                    arr.tofile(fout)
                    total_tokens += len(ids)

    print(f"   ✅ {bin_name}: {total_tokens} токенов, {os.path.getsize(bin_path)//1024**2} МБ")
    return f"OK {bin_name} → {total_tokens} токенов"

def main():
    txt_files = sorted(glob.glob(os.path.join(DATA_TRAIN, "*.txt")))
    if not txt_files:
        print("❌ Нет txt‑файлов в data/train")
        sys.exit(1)

    print(f"📂 Найдено {len(txt_files)} txt‑файлов. Старт предтокенизации…")

    for fp in txt_files:
        result = process_file(fp)
        print(result)

    print("✅ Предтокенизация завершена.")
    print("Теперь можно запускать обучение: make train")

if __name__ == "__main__":
    main()