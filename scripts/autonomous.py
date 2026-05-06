#!/usr/bin/env python3
"""Autonomous overnight work:
1. Wait for pretokenization to complete
2. Process all downloaded data to text
3. Monitor training health (OOM, disk, temps)
4. Report status periodically
"""

import os, sys, time, glob, json, subprocess
from pathlib import Path

PROJECT = "/home/sehaxe/bulba1-python"
HDD = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data"
TRAIN = os.path.join(PROJECT, "data/train")
TMP = os.path.join(HDD, "tmp")
LOG = os.path.join(PROJECT, "logs/autonomous.log")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wait_pretok():
    log("Waiting for pretokenization to complete...")
    while True:
        procs = subprocess.run(["pgrep", "-f", "prepare_full"], capture_output=True, text=True)
        if not procs.stdout.strip():
            break
        txt_count = len(glob.glob(os.path.join(TRAIN, "*.txt")))
        bin_count = len(glob.glob(os.path.join(TRAIN, "*.bin")))
        log(f"  Pretok: {bin_count}/{txt_count} .bin files")
        time.sleep(120)
    log(f"Pretok complete!")
    return len(glob.glob(os.path.join(TRAIN, "*.bin")))


def process_parquets():
    import pyarrow.parquet as pq

    parqs = glob.glob(os.path.join(TMP, "*.parquet")) + glob.glob(os.path.join(TMP, "*.pq"))
    parqs = [p for p in parqs if os.path.getsize(p) > 1_000_000]
    for src in parqs:
        name = os.path.basename(src).replace(".parquet", ".txt").replace(".pq", ".txt")
        dst = os.path.join(TRAIN, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            continue
        try:
            log(f"  Processing {os.path.basename(src)}...")
            pf = pq.ParquetFile(src, memory_map=True)
            with open(dst, "w") as f:
                for batch in pf.iter_batches(batch_size=5000):
                    d = batch.to_pydict()
                    for col in d:
                        if d[col] and isinstance(d[col][0], str):
                            for val in d[col]:
                                if val and len(val.strip()) > 80:
                                    f.write(val.strip()[:100000] + "\n")
                            break
            log(f"    -> {os.path.getsize(dst) / 1024**3:.1f}GB")
        except Exception as e:
            log(f"    FAIL: {e}")


def check_training():
    """Check if training is healthy."""
    log_file = os.path.join(PROJECT, "logs/training_run2.log")
    if not os.path.exists(log_file):
        return True
    with open(log_file) as f:
        content = f.read()

    # Check for OOM
    if "OOM recovery" in content.split("\n")[-10:]:
        log("WARNING: Recent OOM recovery in training log!")

    # Check for GPU temp
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        temp = int(out.stdout.strip())
        if temp > 80:
            log(f"WARNING: GPU temp {temp}°C!")
    except:
        pass

    # Check disk
    disk = os.statvfs(TRAIN)
    free_gb = (disk.f_frsize * disk.f_bavail) / 1024**3
    if free_gb < 5:
        log(f"WARNING: Only {free_gb:.1f}GB free on training disk!")
    elif free_gb < 20:
        log(f"Note: {free_gb:.1f}GB free on training disk")

    return True


def tokenize_new_files():
    """Tokenize any new .txt files that appeared."""
    txt_files = set(glob.glob(os.path.join(TRAIN, "*.txt")))
    bin_files = set(
        os.path.join(TRAIN, os.path.basename(f).replace(".txt", ".bin")) for f in txt_files
    )
    missing = [
        f
        for f in txt_files
        if os.path.join(TRAIN, os.path.basename(f).replace(".txt", ".bin"))
        not in [b for b in glob.glob(os.path.join(TRAIN, "*.bin"))]
    ]
    missing = [f for f in missing if os.path.getsize(f) > 100_000]

    if not missing:
        return

    log(f"Tokenizing {len(missing)} new files...")
    for path in missing:
        out = path.replace(".txt", ".bin")
        sys.path.insert(0, PROJECT)
        from bulba1.data.tokenizer import FastTokenizer
        import struct

        tokenizer = FastTokenizer(os.path.join(PROJECT, "data/tokenizer_fast.json"))
        tokenizer.load()
        ntokens = 0
        try:
            with (
                open(path, "r", encoding="utf-8", errors="ignore") as f_in,
                open(out, "wb") as f_out,
            ):
                for chunk in iter(lambda: f_in.read(8_000_000), ""):
                    substrs = [chunk[i : i + 100000] for i in range(0, len(chunk), 100000)]
                    for ids in tokenizer.encode_batch(substrs):
                        f_out.write(struct.pack(f"<{len(ids)}i", *ids))
                        ntokens += len(ids)
            log(f"  {os.path.basename(path)}: {ntokens / 1e6:.1f}M tokens")
        except Exception as e:
            log(f"  FAIL {os.path.basename(path)}: {e}")


# ── Main autonomous loop ──
log("=== Autonomous mode started ===")
log(f"Training dir: {TRAIN}")
log(f"HDD temp dir: {TMP}")

# Phase 1: Wait for pretok, process new data
bin_count = wait_pretok()
log(f"Phase 1 done: {bin_count} .bin files ready")

# Phase 2: Process any downloaded parquets
log("Phase 2: Processing downloaded data...")
process_parquets()

# Phase 3: Tokenize new files
log("Phase 3: Tokenizing new files...")
tokenize_new_files()

# Phase 4: Monitor loop
log("Phase 4: Monitoring...")
while True:
    check_training()
    bin_count = len(glob.glob(os.path.join(TRAIN, "*.bin")))
    txt_count = len(glob.glob(os.path.join(TRAIN, "*.txt")))

    # Count tokens
    total_tokens = 0
    for bf in glob.glob(os.path.join(TRAIN, "*.bin")):
        total_tokens += os.path.getsize(bf) // 4

    log(f"Status: {bin_count} .bin files, {total_tokens / 1e9:.1f}B tokens")

    # Check if new .txt files appeared
    tokenize_new_files()

    time.sleep(600)  # check every 10 minutes
