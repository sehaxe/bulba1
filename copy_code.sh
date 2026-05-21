#!/bin/bash
# Скрипт для копирования всего кода проекта в буфер обмена

set -e

echo "📋 Копирование кода проекта Bulba1..."

# Функция для вывода содержимого файла
print_file() {
    local file=$1
    echo "=========================================="
    echo "📄 FILE: $file"
    echo "=========================================="
    cat "$file"
    echo ""
    echo ""
}

# Собираем весь код
{
  echo "=== BULBA1 PROJECT CODE DUMP ==="
  echo "Generated at: $(date)"
  echo ""
  
  # Python файлы (исключаем __pycache__, .venv, backups)
  find bulba1 -name "*.py" -type f | grep -v __pycache__ | sort | while read -r file; do
      print_file "$file"
  done
  
  # Конфиги
  find configs -name "*.yaml" -type f | sort | while read -r file; do
      print_file "$file"
  done
  
  # Скрипты
  find scripts -name "*.py" -type f | sort | while read -r file; do
      print_file "$file"
  done
  
  # Telegram бот
  find telegram_bot -name "*.py" -type f | sort | while read -r file; do
      print_file "$file"
  done
  
  # Тесты
  find tests -name "*.py" -type f | sort | while read -r file; do
      print_file "$file"
  done
  
  # Инструменты
  find tools -name "*.py" -type f | sort | while read -r file; do
      print_file "$file"
  done
  
  # Корневые файлы
  for file in cli.py chat.py pyproject.toml Makefile README.md; do
      if [ -f "$file" ]; then
          print_file "$file"
      fi
  done
  
} | tee /tmp/bulba1_code_dump.txt

# Копируем в буфер обмена
if command -v wl-copy &> /dev/null; then
    # Wayland
    cat /tmp/bulba1_code_dump.txt | wl-copy
    echo "✅ Код скопирован в буфер обмена (wl-copy)"
elif command -v xclip &> /dev/null; then
    # X11
    cat /tmp/bulba1_code_dump.txt | xclip -selection clipboard
    echo "✅ Код скопирован в буфер обмена (xclip)"
elif command -v pbcopy &> /dev/null; then
    # macOS
    cat /tmp/bulba1_code_dump.txt | pbcopy
    echo "✅ Код скопирован в буфер обмена (pbcopy)"
else
    echo "⚠️  Не найдена утилита для копирования (wl-copy/xclip/pbcopy)"
    echo "📄 Код сохранен в /tmp/bulba1_code_dump.txt"
fi

# Статистика
echo ""
echo "📊 Статистика:"
echo "   Файлов: $(find bulba1 configs scripts telegram_bot tests tools -name "*.py" -o -name "*.yaml" | wc -l)"
echo "   Строк кода: $(wc -l < /tmp/bulba1_code_dump.txt)"
echo "   Размер: $(du -h /tmp/bulba1_code_dump.txt | cut -f1)"
