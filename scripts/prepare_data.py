import os
import glob
import argparse
from pathlib import Path

from datasets import load_dataset


def write_chunks(items, path, max_items=None):
    with open(path, "w", encoding="utf-8") as f:
        count = 0
        for item in items:
            if max_items is not None and count >= max_items:
                break
            text = item.get("text", item.get("article", item.get("content", "")))
            if text and text.strip():
                f.write(text.replace("\n", " ") + "\n")
                count += 1
    return count


def try_load(name, loader, output_dir, filename):
    path = os.path.join(output_dir, filename)
    try:
        print(f"[ ] Downloading {name}...")
        n = loader(path)
        print(f"  Saved {filename} ({n} items)")
        return True
    except Exception as e:
        print(f"  Skipped {name}: {e}")
        return False


def prepare_datasets(output_dir: str = "data/train", max_items_per_source: int = None):
    os.makedirs(output_dir, exist_ok=True)

    limit = int(max_items_per_source) if max_items_per_source else None
    sources = [
        (
            "WikiText-103",
            lambda p: write_chunks(
                load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True),
                p,
                limit,
            ),
            "wikitext.txt",
        ),
        (
            "ArXiv (scientific_papers)",
            lambda p: write_chunks(
                load_dataset("scientific_papers", "arxiv", split="train", streaming=True), p, limit
            ),
            "arxiv.txt",
        ),
        (
            "Wikipedia (en)",
            lambda p: write_chunks(
                load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True),
                p,
                limit,
            ),
            "wiki_en.txt",
        ),
        (
            "Python code (the-stack)",
            lambda p: write_chunks(
                (
                    {"text": x.get("content", x.get("text", ""))}
                    for x in load_dataset(
                        "bigcode/the-stack", data_dir="data/python", split="train", streaming=True
                    )
                ),
                p,
                limit,
            ),
            "python.txt",
        ),
    ]

    success = 0
    for name, loader, filename in sources:
        if try_load(name, loader, output_dir, filename):
            success += 1

    if success == 0:
        print("[WARNING] No datasets loaded. Generating synthetic fallback data...")
        synth_path = os.path.join(output_dir, "synthetic.txt")
        with open(synth_path, "w", encoding="utf-8") as f:
            for i in range(10_000):
                f.write(f"function example_{i}(x) return x * {i} + {i * i} end\n")
        print(f"  Saved synthetic.txt")

    total_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(output_dir, "*.txt")))
    print(f"\nTotal dataset size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Sources loaded: {success}/{len(sources)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/train")
    parser.add_argument(
        "--max-items", type=int, default=None, help="Max items per source (default: no limit)"
    )
    args = parser.parse_args()
    prepare_datasets(args.output_dir, args.max_items)
