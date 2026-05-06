#!/usr/bin/env python3
"""
Smart data collector — downloads top-quality datasets via wget,
processes parquets with memory_map (zero RAM), writes clean text to HDD.
"""

import os, sys, time, subprocess
from pathlib import Path

HDD = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data"
RAW = os.path.join(HDD, "raw")
TMP = os.path.join(HDD, "tmp")
os.makedirs(RAW, exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def wget(url, name, expected_mb=100):
    out = os.path.join(TMP, name)
    if os.path.exists(out) and os.path.getsize(out) > expected_mb * 0.8 * 1024 * 1024:
        print(f"  SKIP {name} (exists)")
        return out
    print(f"  GET {name}...")
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", out, url, "--timeout=600", "--tries=2"],
            check=True,
            timeout=1200,
        )
    except Exception as e:
        print(f"    FAILED: {e}")
        return None
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"    OK: {mb:.0f}MB")
    return out


def process_parquet(parquet_path, out_name, text_col="text", max_rows=2_000_000):
    """Memory-map parquet, extract text, write to .txt"""
    out = os.path.join(RAW, out_name)
    if os.path.exists(out) and os.path.getsize(out) > 10 * 1024 * 1024:
        print(f"  SKIP {out_name} ({os.path.getsize(out) / 1024**3:.1f}GB)")
        return out

    print(f"  Processing {os.path.basename(parquet_path)} -> {out_name}...")
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path, memory_map=True)

    with open(out, "w", encoding="utf-8") as f:
        written = 0
        for batch in pf.iter_batches(batch_size=10000):
            table = batch.to_pydict()
            if text_col not in table:
                for col in table:
                    if (
                        isinstance(table[col], list)
                        and table[col]
                        and isinstance(table[col][0], str)
                    ):
                        text_col = col
                        break
            if text_col not in table:
                continue

            for text in table[text_col]:
                if not text or not isinstance(text, str):
                    continue
                text = text.strip()
                if len(text) < 80 or len(text) > 50000:
                    continue
                f.write(text + "\n")
                written += 1
                if written >= max_rows:
                    break
            if written % 100000 == 0 and written > 0:
                print(f"    {written} docs, {os.path.getsize(out) / 1024**3:.1f}GB")
            if written >= max_rows:
                break

    gb = os.path.getsize(out) / 1024**3
    print(f"    Done: {gb:.1f}GB, {written} docs")
    return out


# ── Top-quality datasets with direct parquet URLs ──
SOURCES = [
    # FineWeb-Edu sample 10BT — highest quality filtered web
    (
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet",
        "fineweb-edu-0.parquet",
        "fineweb_edu.txt",
    ),
    (
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00001.parquet",
        "fineweb-edu-1.parquet",
        "fineweb_edu_2.txt",
    ),
    # StarCoder — Python code
    (
        "https://huggingface.co/datasets/bigcode/starcoderdata/resolve/main/data/python/train-00000-of-00032.parquet",
        "starcoder-python-0.parquet",
        "starcoder_python.txt",
    ),
    # SlimPajama — curated web
    (
        "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/part-00000-of-01024.parquet",
        "slimpajama-0.parquet",
        "slimpajama.txt",
    ),
    # RedPajama — open LLaMA data, CommonCrawl
    (
        "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/data/common_crawl/part-00000-of-00400.parquet",
        "redpajama-cc-0.parquet",
        "redpajama_cc.txt",
    ),
    # OpenOrca — instruction data
    (
        "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/data/train-00000-of-00010.parquet",
        "openorca-0.parquet",
        "openorca.txt",
    ),
]

print("=== Phase 1: Download parquets ===")
downloaded = []
for url, pq_name, _ in SOURCES:
    path = wget(url, pq_name)
    if path:
        downloaded.append((path, pq_name))

print(f"\n=== Phase 2: Process parquets ({len(downloaded)} files) ===")
for (path, pq_name), (_, _, txt_name) in zip(downloaded, SOURCES[: len(downloaded)]):
    process_parquet(path, txt_name)
    os.remove(path)  # free disk space

print("\n=== DONE ===")
total = sum(os.path.getsize(os.path.join(RAW, f)) for f in os.listdir(RAW) if f.endswith(".txt"))
total_parquets = sum(
    os.path.getsize(os.path.join(TMP, f)) for f in os.listdir(TMP) if f.endswith(".parquet")
)
print(f"Text data: {total / 1024**3:.1f} GB")
print(f"Parquets remaining: {total_parquets / 1024**3:.1f} GB (in {TMP})")
