#!/usr/bin/env python3
"""
Quality-first dataset collection for maximum intelligence per parameter.

★★★ FineWeb-Edu ×20: best filtered web, education-scored (~30B tok)
★★★ The Stack v2 ×14: filtered code — Python, JS, TS, C++, Java (~5B tok)
★★★ MetaMathQA: 395K math Q&A, best math reasoning (~0.3B tok)
★★★ ArXiv ×2: scientific papers, logical reasoning (~1B tok)
★★☆ OpenWebMath: filtered math from CommonCrawl (~1B tok)

Total: ~37-40B tokens of S-tier data.
"""

import os, sys, subprocess

HDD = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data"
TMP = os.path.join(HDD, "tmp")
os.makedirs(TMP, exist_ok=True)

SOURCES = []

for i in range(20):
    SOURCES.append(
        (
            f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{i:03d}_{i:05d}.parquet",
            f"fwe-{i:03d}.pq",
            "★★★ FineWeb-Edu",
        )
    )

CODE = [("python", 5), ("javascript", 3), ("typescript", 2), ("cpp", 2), ("java", 2)]
for lang, n in CODE:
    for i in range(n):
        SOURCES.append(
            (
                f"https://huggingface.co/datasets/bigcode/the-stack-v2/resolve/main/data/{lang}/train-{i:05d}-of-00032.parquet",
                f"stackv2-{lang}-{i:02d}.pq",
                f"★★★ TheStack v2 ({lang})",
            )
        )

SOURCES.append(
    (
        "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json",
        "metamathqa.json",
        "★★★ MetaMathQA",
    )
)

for i in range(2):
    SOURCES.append(
        (
            f"https://huggingface.co/datasets/scientific_papers/resolve/main/arxiv/train-{i:05d}-of-00002.parquet",
            f"arxiv-{i}.pq",
            "★★★ ArXiv",
        )
    )

SOURCES.append(
    (
        "https://huggingface.co/datasets/open-web-math/open-web-math/resolve/main/data/train-00000-of-00048.parquet",
        "owm-0.pq",
        "★★☆ OpenWebMath",
    )
)

# ── ★★★ Synthetic distillation — frontier model outputs ──
# Teaches reasoning patterns from billion-param models into million-param ones.
# Extremely high quality per byte. Critical for tiny model intelligence.
SYNTHETIC = [
    (
        "https://huggingface.co/datasets/Roman1111111/gemini-3.1-pro-hard-high-reasoning/resolve/main/full_dataset.jsonl",
        "gemini31pro.jsonl",
        "★★★ Gemini 3.1 Pro distillation",
    ),
    (
        "https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x/resolve/main/opus46_final.jsonl",
        "claude-opus46.jsonl",
        "★★★ Claude Opus 4.6 distillation",
    ),
    (
        "https://huggingface.co/datasets/beyoru/Deepseek-v4-pro-max-distill-1500x/resolve/main/data/train.parquet",
        "deepseek-v4-train.pq",
        "★★★ DeepSeek V4 distillation",
    ),
    (
        "https://huggingface.co/datasets/beyoru/Deepseek-v4-pro-max-distill-1500x/resolve/main/data/train_math.parquet",
        "deepseek-v4-math.pq",
        "★★★ DeepSeek V4 math distillation",
    ),
]
for url, name, desc in SYNTHETIC:
    SOURCES.append((url, name, desc))

print(f"Sources: {len(SOURCES)}")
print(f"  FineWeb-Edu: 20 chunks → ~200GB text → ~30B tok")
print(f"  Code (5 lang): 14 chunks → ~35GB text → ~5B tok")
print(f"  Math: MetaMathQA + OpenWebMath → ~1.3B tok")
print(f"  Science: ArXiv → ~1B tok")
print(f"  Synthetic: Gemini + Claude + DeepSeek distillation → S-tier reasoning")
print(f"  + Existing 2.4B tok = ~40B total")
print(f"  Overtrain 150M = 267:1 | 300M = 133:1 | 500M = 80:1")
print()

down = fail = 0
total = 0
for url, name, desc in SOURCES:
    out = os.path.join(TMP, name)
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        total += os.path.getsize(out)
        down += 1
        continue
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", out, url, "--timeout=300", "--tries=2"],
            check=True,
            timeout=600,
        )
        down += 1
        total += os.path.getsize(out)
        print(f"[{down}/{len(SOURCES)}] {desc} {os.path.getsize(out) / 1024**3:.1f}GB")
    except Exception as e:
        fail += 1
        print(f"[FAIL] {desc}: {e}")

print(f"\nDone. {down} ok, {fail} fail. Total: {total / 1024**3:.0f}GB → ~{total * 0.085:.0f}B tok")
