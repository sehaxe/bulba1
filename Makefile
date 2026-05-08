.ONESHELL:
SHELL := /bin/bash

UV := uv

.PHONY: help install sync data build train test clean install-services profile-mamba profile-kda profile-moe

help:
	@echo "Доступные команды:"
	@echo "  install        - полная установка окружения (cuda + dev)"
	@echo "  sync           - синхронизация базовых зависимостей без cuda"
	@echo "  data           - скачать все датасеты"
	@echo "  build          - собрать датасет и обучить токенизатор"
	@echo "  train          - запустить обучение с configs/default.yaml"
	@echo "  test           - запустить тесты"
	@echo "  clean          - удалить чекпоинты и логи"
	@echo "  install-services - установить systemd сервисы"
	@echo "  profile-mamba  - профилировать Mamba-3 блок"
	@echo "  profile-kda    - профилировать KDA блок"
	@echo "  profile-moe    - профилировать MoE блок"

install:
	$(UV) sync --extra cuda --extra dev

sync:
	$(UV) sync

data:
	$(UV) run python scripts/download_all_datasets.py

build:
	$(UV) run python scripts/build_and_tokenize.py

train:
	$(UV) run python -m bulba1.cli --config configs/default.yaml

test:
	$(UV) run python -m pytest tests/

clean:
	rm -rf checkpoints/* logs/*
	@echo "Чекпоинты и логи очищены."

install-services:
	@mkdir -p ~/.config/systemd/user
	@cp services/*.service ~/.config/systemd/user/
	@systemctl --user daemon-reload
	@echo "✅ Сервисы установлены:"
	@echo "  systemctl --user start bulba1       # запустить тренировку"
	@echo "  systemctl --user start bulba1-bot   # запустить бота"

profile-mamba:
	$(UV) run python experiments/profile_mamba.py --config configs/default.yaml

profile-kda:
	$(UV) run python experiments/profile_kda.py --config configs/default.yaml

profile-moe:
	$(UV) run python experiments/profile_moe.py --config configs/default.yaml