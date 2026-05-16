#!/bin/bash
# Скрипт собирает только указанные ключевые файлы проекта в один текстовый файл.
# Список файлов можно редактировать.

set -e

PROJECT_DIR="${1:-.}"  # путь к корню проекта (где папка bulba1)
OUTPUT_FILE="key_files_dump.txt"

# Список ключевых файлов (относительно корня проекта)
FILES=(
    "bulba1/model/minichat.py"
    "bulba1/model/block.py"
    "bulba1/model/moe.py"
    "bulba1/model/diff_attn.py"
    "bulba1/model/bit_linear.py"
    "bulba1/model/kda.py"
    "bulba1/model/mhc.py"
    "bulba1/model/mamba.py"
    "bulba1/training/engine.py"
    "bulba1/training/optimizer.py"
    "bulba1/training/checkpoint.py"
    "bulba1/training/ema.py"
    "bulba1/training/monitor.py"
    "bulba1/tokenizer.py"
    "configs/default.yaml"
    "scripts/pretokenize.py"
    "bulba1/cli.py"
)

> "$OUTPUT_FILE"

for file in "${FILES[@]}"; do
    full_path="$PROJECT_DIR/$file"
    if [ -f "$full_path" ]; then
        echo "===== $file =====" >> "$OUTPUT_FILE"
        cat "$full_path" >> "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    else
        echo "[WARNING] Файл не найден: $full_path"
    fi
done

echo "Готово! Ключевые файлы сохранены в '$OUTPUT_FILE'."