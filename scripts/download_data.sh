#!/bin/bash
# Download datasets directly to HDD using wget — no Python RAM overhead
set -e

HDD="/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data/raw"
mkdir -p "$HDD"
cd "$HDD"

download() {
    local url="$1" name="$2"
    if [ -f "$name" ] && [ "$(stat -c%s "$name" 2>/dev/null)" -gt 1000000 ]; then
        echo "SKIP $name (exists: $(du -sh "$name" | cut -f1))"
        return
    fi
    echo "GET $name ..."
    wget -q --show-progress -O "$name" "$url" --timeout=300 --tries=3 || echo "FAILED: $name"
}

echo "=== Downloading datasets to HDD ==="
echo "HDD: $HDD ($(df -h "$HDD" | tail -1 | awk '{print $4}') free)"
echo ""

# 1 — TinyStories (GPT-4 generated, high quality)
download "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
        "TinyStoriesV2-GPT4-train.txt"

# 2 — FineWeb-Edu sample 10BT (highest quality filtered web)
download "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet" \
        "fineweb-edu-10bt-0.parquet"

# 3 — StackOverflow + StackExchange Q&A (from Internet Archive)
download "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z" \
        "stackoverflow-Posts.7z"

# 4 — OpenWebText2
download "https://the-eye.eu/public/AI/pile/train/08.jsonl.zst" \
        "pile-08.jsonl.zst"

# 5 — WikiText-103 raw
download "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip" \
        "wikitext-103-raw-v1.zip"

echo ""
echo "=== DONE ==="
du -sh "$HDD"
ls -lh "$HDD"/*.txt "$HDD"/*.parquet "$HDD"/*.7z "$HDD"/*.zst 2>/dev/null
