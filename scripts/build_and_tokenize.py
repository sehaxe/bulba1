#!/usr/bin/env python3
"""
ПОТОКОВАЯ СБОРКА ДАННЫХ БЕЗ OOM – 100% ГАРАНТИЯ
Читает parquet чанками (pyarrow), пишет txt, тренирует токенизатор на txt.
НЕ ИСПОЛЬЗУЕТ shuf, НЕ ЗАГРУЖАЕТ ВСЁ В ПАМЯТЬ.
Если скрипт прервётся, можно перезапустить – готовые txt пропустятся.
"""

import os, sys, glob, json, gzip, subprocess

# Установим pyarrow, если его нет
try:
    import pyarrow.parquet as pq
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    import pyarrow.parquet as pq

TMP = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/bulba1-data/tmp"
DATA_TRAIN = "data/train"
TOKENIZER_PATH = "data/tokenizer_fast.json"

os.makedirs(DATA_TRAIN, exist_ok=True)

def convert_parquet_to_txt(src, dst):
    import gc
    gc.disable()   # экономим время
    import pyarrow.dataset as ds
    dataset = ds.dataset(src, format="parquet")
    total = 0
    with open(dst, "w", encoding="utf-8") as f:
        # 2 миллиона строк за раз – оптимально для 16 ГБ RAM
        for batch in dataset.to_batches(batch_size=2_000_000):
            df = batch.to_pandas()
            for text in df["text"].fillna("").astype(str):
                if text:
                    f.write(text.replace('\n', '\x01') + '\n')
                    total += 1
    gc.enable()
    return total

def convert_jsonl_to_txt(src, dst):
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or obj.get("instruction") or obj.get("output") or ""
                if not text and "messages" in obj:
                    text = "\n".join(m.get("content","") for m in obj["messages"] if m.get("content"))
                if text:
                    fout.write(text.replace('\n', '\x01') + '\n')
            except:
                pass

def convert_json_to_txt(src, dst):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as fout:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    fout.write(item.replace('\n', '\x01') + '\n')
                elif isinstance(item, dict):
                    q = item.get("query", "")
                    a = item.get("response", "") or item.get("answer", "")
                    if q and a:
                        fout.write(f"Q: {q}\nA: {a}".replace('\n', '\x01') + '\n')

def convert_gz_to_txt(src, dst):
    with gzip.open(src, "rt", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    fout.write(text.replace('\n', '\x01') + '\n')
            except:
                pass

def main():
    print("📄 Конвертация сырых файлов в txt ...")
    files = sorted(glob.glob(os.path.join(TMP, '*')))
    for fp in files:
        fname = os.path.basename(fp)
        dst = os.path.join(DATA_TRAIN, fname.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(dst):
            continue
        if fname.endswith('.pq') or fname.endswith('.parquet'):
            print(f"  {fname} (parquet) -> {os.path.basename(dst)}")
            rows = convert_parquet_to_txt(fp, dst)
            print(f"    извлечено {rows} документов")
        elif fname.endswith('.jsonl'):
            print(f"  {fname} (jsonl) -> {os.path.basename(dst)}")
            convert_jsonl_to_txt(fp, dst)
        elif fname.endswith('.json'):
            print(f"  {fname} (json) -> {os.path.basename(dst)}")
            convert_json_to_txt(fp, dst)
        elif fname.endswith('.json.gz'):
            print(f"  {fname} (json.gz) -> {os.path.basename(dst)}")
            convert_gz_to_txt(fp, dst)

    print("\n🔤 Обучение токенизатора на txt-файлах (потоково, <2 ГБ RAM) ...")
    sys.path.insert(0, os.getcwd())
    from bulba1.tokenizer import SmartTokenizer
    txt_files = sorted(glob.glob(os.path.join(DATA_TRAIN, "*.txt")))
    if not txt_files:
        print("❌ Нет txt-файлов")
        sys.exit(1)

    tok = SmartTokenizer(
        vocab_size=None,
        model_path=TOKENIZER_PATH,
        target_params=150_000_000,
        auto_detect=True,
        sample_size=10_000_000,   # только 10 МБ для анализа!
    )
    tok.train(txt_files)
    print(tok.get_analysis_report())
    print(f"✅ Токенизатор сохранён: {TOKENIZER_PATH}")
    print("\n🎉 ВСЁ ГОТОВО! Запускайте обучение: make train")

if __name__ == "__main__":
    main()