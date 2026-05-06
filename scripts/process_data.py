#!/usr/bin/env python3
"""Process all downloaded parquets + JSON → clean .txt files.
Memory-efficient: uses pyarrow memory_map to avoid RAM pressure."""

import os, sys, json, glob

HDD = os.environ.get("BULBA1_DATA", "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data")
TRAIN = os.environ.get("BULBA1_TRAIN", "data/train")
TMP = os.path.join(HDD, "tmp") if os.path.exists(HDD) else "data/tmp"
RAW = os.path.join(HDD, "raw") if os.path.exists(HDD) else "data/raw"

if os.path.exists(HDD):
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)


def extract_parquet(src, dst_name, max_gb=15):
    dst = os.path.join(TRAIN, dst_name)
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
        print(f"  SKIP {dst_name} ({os.path.getsize(dst) / 1024**3:.1f}GB)")
        return
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(src, memory_map=True)
    except Exception as e:
        print(f"  FAIL {src}: {e}")
        return

    written = 0
    max_bytes = max_gb * 1024**3
    with open(dst, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=5000):
            d = batch.to_pydict()
            col = "text" if "text" in d else d.keys()
            if not isinstance(col, str):
                for c in d:
                    if d[c] and isinstance(d[c][0], str):
                        col = c
                        break
            if not isinstance(col, str):
                continue
            for val in d[col]:
                if not val or not isinstance(val, str):
                    continue
                val = val.strip()
                if len(val) < 80 or len(val) > 100000:
                    continue
                f.write(val + "\n")
                written += len(val.encode("utf-8"))
                if written >= max_bytes:
                    break
            if written >= max_bytes:
                break
    print(f"  {dst_name}: {os.path.getsize(dst) / 1024**3:.1f}GB")


def extract_metamathqa(src, dst_name="metamathqa.txt"):
    dst = os.path.join(TRAIN, dst_name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"  SKIP {dst_name} (exists)")
        return
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as out:
        count = 0
        for item in data:
            q = item.get("query", item.get("question", item.get("instruction", "")))
            a = item.get("response", item.get("answer", item.get("output", "")))
            if q and a:
                out.write(f"Q: {q}\nA: {a}\n\n")
                count += 1
            if count % 50000 == 0:
                print(f"  {count} QA pairs...")
    print(f"  {dst_name}: {os.path.getsize(dst) / 1024**3:.1f}GB, {count} pairs")


print("=== Processing parquets → text ===")
for pq_path, txt_name in [
    ("fineweb-edu-000.parquet", "fineweb_edu_00.txt"),
    ("fineweb-000.parquet", "fineweb_00.txt"),
    ("fwe-000.pq", "fwe_00.txt"),
]:
    src = os.path.join(TMP, pq_path)
    if os.path.exists(src) and os.path.getsize(src) > 1000000:
        extract_parquet(src, txt_name)

print("\n=== Processing MetaMathQA ===")
src = os.path.join(TMP, "metamathqa.json")
if os.path.exists(src):
    extract_metamathqa(src)

print("\n=== Copy existing raw files to train dir ===")
for f in glob.glob(os.path.join(RAW, "*.txt")):
    name = os.path.basename(f)
    dst = os.path.join(TRAIN, name)
    if not os.path.exists(dst) and os.path.getsize(f) > 100000:
        import shutil

        shutil.copy2(f, dst)
        print(f"  + {name}")

print(
    f"\nDone. Train dir: {sum(os.path.getsize(os.path.join(TRAIN, f)) for f in os.listdir(TRAIN) if f.endswith('.txt')) / 1024**3:.1f}GB"
)
