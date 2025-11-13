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

serve:
	credit-risk serve --config $(CONFIG)

monitor-demo:
	credit-risk simulate --n 5000 --drift 1.2 --out data/raw/drifted_batch.csv
	credit-risk monitor --config $(CONFIG) --input data/raw/drifted_batch.csv

runs:
	credit-risk runs --config $(CONFIG)

registry:
	credit-risk registry list --config $(CONFIG)

docker:
	docker compose up --build

clean:
	rm -rf data/interim data/processed artifacts reports logs .pipeline_cache mlruns_local model_registry
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
