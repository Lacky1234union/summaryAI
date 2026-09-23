.DEFAULT_GOAL := help
.PHONY: help install lint fmt test test-all bench-asr run clean

help:                 ## Показать эти команды
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Окружение ───────────────────────────────────────────────────────────────

install:              ## Поставить зависимости (uv sync, включая dev-группу)
	uv sync

# ── Гейт ────────────────────────────────────────────────────────────────────

lint:                 ## Линтер и типы: ruff check + ruff format --check + mypy
	uv run ruff check summuraze tests
	uv run ruff format --check summuraze tests
	uv run mypy summuraze

fmt:                  ## Отформатировать и починить, что чинится автоматически
	uv run ruff check --fix summuraze tests
	uv run ruff format summuraze tests

test:                 ## Быстрые тесты (без slow и eval)
	uv run pytest

test-all:             ## Все тесты, включая slow и eval (нужны модели и фикстуры)
	uv run pytest -m ""

# ── Работа ──────────────────────────────────────────────────────────────────

run:                  ## Прогнать конвейер: make run FILE=запись.mp4
	@test -n "$(FILE)" || (echo "Укажи FILE: make run FILE=lecture.mp4" && exit 1)
	uv run summuraze run "$(FILE)"

bench-asr:            ## Бенчмарк STT-моделей на этом железе (#9) → docs/BENCHMARKS.md
	uv run summuraze bench asr

clean:                ## Убрать кэши инструментов и артефакты сборки
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
