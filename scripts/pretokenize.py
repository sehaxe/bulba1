import os, sys, glob, struct, multiprocessing as mp
from functools import partial
from tqdm import tqdm

sys.path.insert(0, "/home/sehaxe/bulba1-python")
from bulba1.data.tokenizer import FastTokenizer

BATCH_BYTES = 8_000_000
CHUNK_BYTES = 100_000


def _process_one(path: str, model_path: str) -> tuple:
    out_path = path + ".bin"
    fsize = os.path.getsize(path)
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(path):
        if os.path.getsize(out_path) >= fsize * 0.2:
            return (path, 0, True)

    tokenizer = FastTokenizer(model_path=model_path)
    tokenizer.load()
    ntokens = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f_in, open(out_path, "wb") as f_out:
        for text_chunk in iter(lambda: f_in.read(BATCH_BYTES), ""):
            substrs = [
                text_chunk[i : i + CHUNK_BYTES] for i in range(0, len(text_chunk), CHUNK_BYTES)
            ]
            for ids in tokenizer.encode_batch(substrs):
                f_out.write(struct.pack(f"<{len(ids)}i", *ids))
                ntokens += len(ids)
    return (path, ntokens, False)


def pretokenize(data_dir: str = "data/train", model_path: str = "data/tokenizer_fast.json"):
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True))
    big_files = [f for f in files if os.path.getsize(f) > 100_000]
    if not big_files:
        print("No .txt files found")
        return

    workers = max(1, min(4, mp.cpu_count() - 1))
    total_bytes = sum(os.path.getsize(f) for f in big_files)
    total_tokens = 0
    skipped = 0

    print(
        f"Pre-tokenizing {len(big_files)} files ({total_bytes / 1024**3:.1f} GB) with {workers} workers..."
    )
    pbar = tqdm(
        total=total_bytes, desc="Tokenizing", unit="B", unit_scale=True, position=0, leave=True
    )

    fn = partial(_process_one, model_path=model_path)
    with mp.Pool(processes=workers) as pool:
        for path, n_tok, was_skipped in pool.imap_unordered(fn, big_files):
            fsize = os.path.getsize(path)
            pbar.update(fsize)
            pbar.refresh()
            name = os.path.basename(path)
            if was_skipped:
                skipped += 1
            else:
                total_tokens += n_tok
                pbar.write(f"  {name}: {n_tok / 1e6:.1f}M tokens")

    pbar.close()
    print(f"Done. Skipped {skipped} files. Total tokens: {total_tokens / 1e6:.1f}M")


if __name__ == "__main__":
    pretokenize()
