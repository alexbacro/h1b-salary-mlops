# Convenience targets. Run `make help` to list them.
PYTHON ?= python

.PHONY: help setup lint format test ingest clean-data split baseline register-baseline data drift

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install the package and dev dependencies into the active environment
	$(PYTHON) -m pip install -r requirements-dev.txt -e .

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## Auto-fix lint issues and format code
	ruff check --fix .
	ruff format .

test:  ## Run unit tests
	$(PYTHON) -m pytest

ingest:  ## data/raw/*.xlsx -> data/interim/*.parquet
	$(PYTHON) -m h1b.ingest

clean-data:  ## data/interim -> data/processed/lca_clean.parquet (+ reports/ingest_report.json)
	$(PYTHON) -m h1b.clean

split:  ## Temporal train/val/test split (test goes to data/holdout/)
	$(PYTHON) -m h1b.split

baseline:  ## Median-by-SOC-and-state baseline, evaluated on val
	$(PYTHON) -m h1b.baseline

data: ingest clean-data split baseline  ## Run the whole data stage end to end

drift:  ## Evidently drift reports: train vs val (features and target) -> reports/*.html
	$(PYTHON) -m h1b.drift_report

register-baseline:  ## Register the median baseline in MLflow as h1b-salary-model@champion
	$(PYTHON) -m h1b.register_baseline
