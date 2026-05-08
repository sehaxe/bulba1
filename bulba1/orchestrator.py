#!/usr/bin/env python3
"""
Умный оркестратор Bulba‑1: полный цикл от загрузки данных до обучения.
Запускает скрипты download и build, затем передаёт управление cli.py.
"""

import subprocess
import sys
from pathlib import Path

class BulbaOrchestrator:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path

    def download(self):
        print("📥 Скачивание всех датасетов...")
        script = Path("scripts/download_all_datasets.py")
        if not script.exists():
            print("❌ Скрипт download_all_datasets.py не найден.")
            sys.exit(1)
        subprocess.run([sys.executable, str(script)], check=True)

    def build(self):
        print("🔤 Сборка датасета и обучение токенизатора...")
        script = Path("scripts/build_and_tokenize.py")
        if not script.exists():
            print("❌ Скрипт build_and_tokenize.py не найден.")
            sys.exit(1)
        subprocess.run([sys.executable, str(script)], check=True)

    def train(self):
        print("🚀 Запуск тренировки...")
        cmd = [sys.executable, "-m", "bulba1.cli"]
        if self.config_path:
            cmd += ["--config", self.config_path]
        subprocess.run(cmd, check=True)

    def run_full(self, skip_download=False, skip_build=False):
        if not skip_download:
            self.download()
        if not skip_build:
            self.build()
        self.train()