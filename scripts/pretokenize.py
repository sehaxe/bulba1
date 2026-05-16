#!/usr/bin/env python3
"""
САМАЯ СТАБИЛЬНАЯ ПРЕДТОКЕНИЗАЦИЯ + YAML + JSONL ЛОГИ
====================================================================
Работает в 1 процесс, но использует 100% ядер процессора за счет Rust.
Идеально сохраняет переносы строк, не падает по памяти.
Выводит логи для Telegram-бота в формате .jsonl.
"""

import argparse
import array
import glob
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import yaml  # Заменили json на yaml для манифеста
from tqdm import tqdm

# Оставляем параллелизм Rust включенным на максимум!
os.environ["TOKENIZERS_PARALLELISM"] = "true"

try:
    from bulba1.tokenizer import FastTokenizer
except ImportError:
    print("WARNING: bulba1.tokenizer not found.")
    FastTokenizer = None

# Читаем стабильными кусками по 50 Мегабайт
CHUNK_SIZE_BYTES = 50 * 1024 * 1024

# ==============================================================================
# Helpers
# ==============================================================================

def guess_domain(filename: str) -> str:
    name = os.path.basename(filename).lower()
    while name.endswith('.txt') or name.endswith('.json'):
        name = name.rsplit('.', 1)[0]
    if 'math' in name or 'metamathqa' in name:
        return 'math'
    base = name.rstrip('0123456789-_')
    prefix = base.split('-')[0].split('_')[0]
    return prefix if prefix else 'unknown'

def sample_indices(n_total: int, n_take: int, seed: int):
    rng = np.random.default_rng(seed)
    if n_take >= n_total:
        return rng.permutation(n_total)
    return rng.choice(n_total, size=n_take, replace=False)

def log_to_jsonl(log_path: str, record: dict):
    """Записывает 1 строчку JSONL для Telegram бота"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ==============================================================================
# Phase 1: Parallel Tokenization (Stable CPU version)
# ==============================================================================

def tokenize_phase(
    data_dir: str,
    tmp_dir: str,
    tokenizer,
    seq_len: int,
    stride: int,
    manifest_path: str,
    log_file: str,
    domain_weights: dict[str, float] | None = None,
):
    txt_files = sorted(glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True))
    if not txt_files:
        print(f"❌ Нет txt‑файлов в {data_dir}")
        sys.exit(1)

    # Читаем YAML манифест
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        existing_files = {entry['path'] for entry in manifest.get('files',[])}
        if not domain_weights and 'domain_weights' in manifest:
            domain_weights = manifest['domain_weights']
    else:
        manifest = {
            'version': 1, 'seq_len': seq_len, 'stride': stride,
            'domain_weights': domain_weights if domain_weights else {}, 'files':[],
        }
        existing_files = set()

    remaining_files =[f for f in txt_files if os.path.relpath(f, start=data_dir) not in existing_files]
    if not remaining_files:
        print("✅ Все файлы уже токенизированы. Пропускаем.")
        return

    print(f"📂 Найдено {len(remaining_files)} новых файлов. Старт стабильной токенизации…")
    os.makedirs(tmp_dir, exist_ok=True)

    new_files_stats =[]

    # Глобальные счетчики для логов бота
    total_bytes_all = sum(os.path.getsize(f) for f in remaining_files)
    processed_bytes_all = 0
    total_tokens_all = sum(e.get('tokens', 0) for e in manifest.get('files',[]))

    # Сбрасываем (очищаем) лог файл перед новым запуском
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    open(log_file, "w").close()

    for txt_path in remaining_files:
        rel_path = os.path.relpath(txt_path, start=data_dir)
        domain = guess_domain(os.path.basename(txt_path))

        domain_bin_path = os.path.join(tmp_dir, f".{domain}.tok.bin")
        file_size_bytes = os.path.getsize(txt_path)

        print(f"\n🔤 Файл: {os.path.basename(txt_path)} ({file_size_bytes / 1024**2:.1f} МБ)")

        file_tokens = 0
        processed_file_bytes = 0

        last_log_time = time.time()
        bytes_since_log = 0
        tokens_since_log = 0

        with open(txt_path, "rb") as fin, \
             open(domain_bin_path, "ab") as fout, \
             tqdm(total=file_size_bytes, unit='B', unit_scale=True, unit_divisor=1024) as pbar:

            while True:
                # Читаем строками, чтобы не рвать слова (ровно ~50 МБ за раз)
                raw_lines = fin.readlines(CHUNK_SIZE_BYTES)
                if not raw_lines:
                    break

                # Декодируем
                str_lines =[line.decode("utf-8", errors="ignore") for line in raw_lines]

                # Токенизируем (Rust съедает это параллельно на всех ядрах)
                encoded = tokenizer.encode_batch(str_lines)

                # Сплющиваем массив
                flattened_ids = itertools.chain.from_iterable(encoded)
                bin_array = array.array('i', flattened_ids)

                # Записываем на диск монолитом
                bin_array.tofile(fout)

                # Обновляем метрики
                chunk_bytes = sum(len(line) for line in raw_lines)
                chunk_tokens = len(bin_array)

                file_tokens += chunk_tokens
                total_tokens_all += chunk_tokens
                processed_file_bytes += chunk_bytes
                processed_bytes_all += chunk_bytes

                bytes_since_log += chunk_bytes
                tokens_since_log += chunk_tokens

                pbar.update(chunk_bytes)
                pbar.set_postfix_str(f"Токены: {file_tokens / 1e6:.2f}M")

                # Логгирование в .jsonl каждую секунду (для Telegram бота)
                now = time.time()
                if now - last_log_time >= 1.0:
                    dt = now - last_log_time
                    mb_per_sec = (bytes_since_log / 1024 / 1024) / dt
                    tok_per_sec = tokens_since_log / dt

                    file_pct = (processed_file_bytes / file_size_bytes) * 100 if file_size_bytes else 100
                    total_pct = (processed_bytes_all / total_bytes_all) * 100 if total_bytes_all else 100

                    log_to_jsonl(log_file, {
                        "task": "pretokenize",
                        "current_file": os.path.basename(txt_path),
                        "file_progress_pct": round(file_pct, 1),
                        "total_progress_pct": round(total_pct, 1),
                        "mb_per_sec": round(mb_per_sec, 2),
                        "tok_per_sec": int(tok_per_sec),
                        "file_tokens_m": round(file_tokens / 1e6, 2),
                        "total_tokens_m": round(total_tokens_all / 1e6, 2),
                        "timestamp": int(now)
                    })

                    bytes_since_log = 0
                    tokens_since_log = 0
                    last_log_time = now

        new_files_stats.append({
            'path': rel_path,
            'domain': domain,
            'tokens': file_tokens,
            'enabled': True
        })

    # Сохраняем YAML манифест
    manifest['files'].extend(new_files_stats)
    if not manifest['domain_weights']:
        domains_in_data = {entry['domain'] for entry in manifest['files'] if entry['enabled']}
        if domains_in_data:
            manifest['domain_weights'] = {d: 1.0 / len(domains_in_data) for d in domains_in_data}
        else:
            manifest['domain_weights'] = {'unknown': 1.0}

    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n✅ Токенизация завершена! Данные сохранены в манифест: {manifest_path}")


# ==============================================================================
# Phase 2: Balance & Shard
# ==============================================================================

def balance_phase(
    manifest_path: str,
    tmp_dir: str,
    output_dir: str,
    num_shards: int,
    seed: int,
    keep_tmp: bool,
) -> list[str]:
    with open(manifest_path, encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    seq_len = manifest['seq_len']
    stride = manifest['stride']
    files = manifest['files']

    domain_tokens = defaultdict(int)
    for entry in files:
        if entry['enabled']:
            domain_tokens[entry['domain']] += entry['tokens']

    domain_samples = {}
    for d, tokens in domain_tokens.items():
        if tokens >= seq_len + 1:
            domain_samples[d] = (tokens - (seq_len + 1)) // stride + 1
        else:
            domain_samples[d] = 0

    active_domains = [d for d in domain_samples if domain_samples[d] > 0]
    if not active_domains:
        raise RuntimeError("No active domains found. Check if tokenization phase produced any data.")

    # Берём 100% данных
    take = {}
    total_picks = 0
    for d in active_domains:
        take[d] = domain_samples[d]
        total_picks += take[d]

    print("\n📊 Состав датасета (используется 100% доступных данных):")
    for d in active_domains:
        share = (take[d] / total_picks) * 100
        print(f"  {d}: {take[d]:,} семплов ({share:.1f}%)")
    print(f"  ИТОГО: {total_picks:,} семплов")

    # Случайные перестановки индексов внутри каждого домена
    domain_indices = {}
    for d in active_domains:
        domain_indices[d] = sample_indices(domain_samples[d], take[d], seed + hash(d) % 100000)

    sample_size = seq_len + 1
    os.makedirs(output_dir, exist_ok=True)

    shard_paths = []
    BUF_SIZE = 50000

    print(f"\n📦 Пишем {total_picks:,} семплов в {num_shards} шард(ов)...")

    domain_mmaps = {}
    for d in active_domains:
        path = os.path.join(tmp_dir, f".{d}.tok.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing domain binary: {path}")
        domain_mmaps[d] = np.memmap(path, dtype=np.int32, mode='r')

    # Генерируем массив позиций для глобального перемешивания
    # Используем np.int64 для sample_idx, чтобы исключить переполнение
    positions = np.empty(total_picks, dtype=[('domain_idx', np.int8), ('sample_idx', np.int64)])
    pos = 0
    for di, d in enumerate(active_domains):
        cnt = len(domain_indices[d])
        positions['domain_idx'][pos:pos+cnt] = di
        positions['sample_idx'][pos:pos+cnt] = domain_indices[d]
        pos += cnt

    # Глобальный шаффл всех доменов между собой
    rng = np.random.default_rng(seed)
    rng.shuffle(positions)

    from contextlib import ExitStack

    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(open(os.path.join(output_dir, f"train_balanced_{i:03d}.bin"), 'wb'))
            for i in range(num_shards)
        ]
        shard_buf_idx = [0] * num_shards
        shard_bufs = [np.empty((BUF_SIZE, sample_size), dtype=np.int32) for _ in range(num_shards)]

        for i in tqdm(range(total_picks), desc="Writing shards"):
            di = positions['domain_idx'][i]
            sample_idx = positions['sample_idx'][i]
            shard_idx = i % num_shards

            mm = domain_mmaps[active_domains[di]]
            # Главное исправление – явный Python int
            offset = int(sample_idx) * stride

            shard_bufs[shard_idx][shard_buf_idx[shard_idx]] = mm[offset:offset + sample_size]
            shard_buf_idx[shard_idx] += 1

            if shard_buf_idx[shard_idx] == BUF_SIZE:
                shard_bufs[shard_idx].tofile(shard_files[shard_idx])
                shard_buf_idx[shard_idx] = 0

        # Сбрасываем остатки буферов
        for shard_idx in range(num_shards):
            if shard_buf_idx[shard_idx] > 0:
                shard_bufs[shard_idx][:shard_buf_idx[shard_idx]].tofile(shard_files[shard_idx])
            shard_paths.append(os.path.join(output_dir, f"train_balanced_{shard_idx:03d}.bin"))

    for mm in domain_mmaps.values():
        mm._mmap.close()

    if not keep_tmp:
        for d in active_domains:
            path = os.path.join(tmp_dir, f".{d}.tok.bin")
            if os.path.exists(path):
                os.remove(path)

    # Обновляем манифест финальной статистикой
    manifest['balanced_output'] = {
        'shards': [os.path.relpath(p, start=output_dir) for p in shard_paths],
        'total_samples': total_picks,
        'seed': seed,
        'proportions': {d: round((take[d]/total_picks)*100, 2) for d in active_domains}
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return shard_paths

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bulba1 preprocessing")
    parser.add_argument('--data-dir', type=str, default='data/train')
    parser.add_argument('--tokenizer-path', type=str, default='data/tokenizer_fast.json')
    parser.add_argument('--output-dir', type=str, default='data/tokenized')
    # Изменили default на .yaml
    parser.add_argument('--manifest', type=str, default='data_manifest.yaml')
    # Путь куда будут писаться логи для бота
    parser.add_argument('--log-file', type=str, default='logs/pretokenize.jsonl')
    parser.add_argument('--tmp-dir', type=str, default=None)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', choices=['tokenize', 'balance', 'all'], default='all')
    parser.add_argument('--keep-tmp', action='store_true')
    parser.add_argument('--domain-weights', type=str, default=None)

    # Игнорируется, так как используем Rust Multi-threading
    parser.add_argument('--num-tokenizer-workers', type=int, default=0, help="Ignored")

    args = parser.parse_args()

    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, '.tmp_domains')
    domain_weights = json.loads(args.domain_weights) if args.domain_weights else None

    if FastTokenizer is None:
        sys.exit(1)

    tokenizer = FastTokenizer(args.tokenizer_path)
    tokenizer.load()

    if args.phase in ('tokenize', 'all'):
        tokenize_phase(
            data_dir=args.data_dir,
            tmp_dir=tmp_dir,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            manifest_path=args.manifest,
            log_file=args.log_file,
            domain_weights=domain_weights,
        )

    if args.phase in ('balance', 'all'):
        if not os.path.exists(args.manifest):
            print("❌ ОШИБКА: Манифест не найден. Сначала запустите фазу tokenize.")
            sys.exit(1)
        balance_phase(
            manifest_path=args.manifest,
            tmp_dir=tmp_dir,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            seed=args.seed,
            keep_tmp=args.keep_tmp,
        )

if __name__ == "__main__":
    main()


