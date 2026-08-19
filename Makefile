.PHONY: install lint format typecheck test check prepare-data train evaluate

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	black --check .

typecheck:
	mypy .

test:
	pytest -q

check: lint format typecheck test

prepare-data:
	python -m src.cli prepare-data

train:
	python -m src.cli train

evaluate:
	python -m src.cli evaluate
