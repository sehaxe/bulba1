#!/usr/bin/env python3
"""Быстрая предтокенизация: батчи, encode_batch, memmap."""
import os, sys, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from bulba1.tokenizer import FastTokenizer

DATA_TRAIN = "data/train"
TOKENIZER_PATH = "data/tokenizer_fast.json"
BIN_OUT = "data/tokenized"
CHUNK_LINES = 100_000

os.makedirs(BIN_OUT, exist_ok=True)
tok = FastTokenizer(TOKENIZER_PATH)
tok.load()

def process_file(txt_path):
    bin_path = os.path.join(BIN_OUT, os.path.basename(txt_path).replace(".txt", ".bin"))
    if os.path.exists(bin_path):
        return f"SKIP {os.path.basename(txt_path)}"
    est_size = os.path.getsize(txt_path) // 10 * 4
    arr = np.memmap(bin_path, dtype=np.int32, mode='w+', shape=(est_size // 4,))
    write_pos = 0
    with open(txt_path, encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) >= CHUNK_LINES:
                encoded = tok.encode_batch(batch)
                for ids in encoded:
                    if ids:
                        end = write_pos + len(ids)
                        if end > arr.shape[0]:
                            arr = np.memmap(bin_path, dtype=np.int32, mode='r+', shape=(end,))
                        arr[write_pos:end] = ids
                        write_pos = end
                batch = []
        if batch:
            encoded = tok.encode_batch(batch)
            for ids in encoded:
                if ids:
                    end = write_pos + len(ids)
                    if end > arr.shape[0]:
                        arr = np.memmap(bin_path, dtype=np.int32, mode='r+', shape=(end,))
                    arr[write_pos:end] = ids
                    write_pos = end
    if write_pos < arr.shape[0]:
        arr = np.memmap(bin_path, dtype=np.int32, mode='r+', shape=(write_pos,))
    arr.flush()
    return f"OK {os.path.basename(txt_path)} → {os.path.basename(bin_path)} ({write_pos} токенов)"

def main():
    txt_files = sorted(glob.glob(os.path.join(DATA_TRAIN, "*.txt")))
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = executor.map(process_file, txt_files)
    for r in results:
        print(r)
    print("✅ Предтокенизация завершена. Переключите data_dir на 'data/tokenized' в конфиге.")

if __name__ == "__main__":
    main()