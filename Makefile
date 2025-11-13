CONFIG ?= configs/pipeline.yaml

.PHONY: install lint format test pipeline serve monitor-demo runs registry docker clean

install:
	pip install -e ".[dev,lightgbm]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

pipeline:
	credit-risk run --config $(CONFIG)
