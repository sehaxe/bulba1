#!/usr/bin/env python3
"""Stream datasets from HuggingFace directly to HDD — zero RAM pressure."""

import os, sys, time

HDD = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data"
RAW = os.path.join(HDD, "raw")
os.makedirs(RAW, exist_ok=True)


def stream_hf(name, config=None, split="train", max_gb=20):
    out = os.path.join(RAW, f"{name.replace('/', '_')}.txt")
    if os.path.exists(out) and os.path.getsize(out) > max_gb * 0.7 * 1024**3:
        print(f"  SKIP {name} ({os.path.getsize(out) / 1024**3:.1f}GB)")
        return out

    print(f"  DOWNLOAD {name} (max {max_gb}GB)...")
    from datasets import load_dataset

    try:
        kw = {"path": name, "split": split, "streaming": True}
        if config:
            kw["name"] = config
        ds = load_dataset(**kw)
    except Exception as e:
        print(f"    FAILED: {e}")
        return None

    written = 0
    max_bytes = max_gb * 1024**3
    last_report = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", item.get("content", item.get("prompt", "")))
            if not text or not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < 80 or len(text) > 50000:
                continue
            f.write(text + "\n")
            written += len(text.encode("utf-8"))
            if time.time() - last_report > 15:
                print(f"    {written / 1024**3:.1f}GB...")
                last_report = time.time()
            if written >= max_bytes:
                break
    gb = os.path.getsize(out) / 1024**3
    print(f"    DONE {gb:.1f}GB → {out}")
    return out


def stream_hf_subsets(name, configs, max_gb_per=3):
    for cfg in configs:
        stream_hf(name, cfg, max_gb=max_gb_per)


# ── Execute ──

print("=== Streaming datasets to HDD ===")
start = time.time()

# 1. FineWeb — high-quality filtered CommonCrawl (already have 2.4GB)
stream_hf("HuggingFaceFW/fineweb", "sample-10BT", max_gb=20)

# 2. TinyStories — simple coherent text (already have 1GB)
stream_hf("roneneldan/TinyStories", max_gb=2)

# 3. MathInstruct — math reasoning (already have 172MB, get more)
stream_hf("TIGER-Lab/MathInstruct", max_gb=2)

# 4. SlimPajama — curated web corpus
stream_hf("cerebras/SlimPajama-627B", max_gb=15)

# 5. RedPajama — open LLaMA data reproduction
stream_hf("togethercomputer/RedPajama-Data-1T", "common_crawl", max_gb=5)
stream_hf("togethercomputer/RedPajama-Data-1T", "c4", max_gb=3)

# 6. StarCoder — code
stream_hf("bigcode/starcoderdata", "python", max_gb=5)
stream_hf("bigcode/starcoderdata", "javascript", max_gb=2)

# 7. Wiki (already have, skip)

# 8. ArXiv abstracts — scientific
stream_hf("scientific_papers", "arxiv", max_gb=3)

# 9. OpenOrca — instruction data
stream_hf("Open-Orca/OpenOrca", max_gb=3)

elapsed = time.time() - start
total = sum(os.path.getsize(os.path.join(RAW, f)) for f in os.listdir(RAW) if f.endswith(".txt"))
print(f"\n=== DONE in {elapsed / 60:.0f}m ===")
print(f"Total raw text: {total / 1024**3:.1f} GB")
