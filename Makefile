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

# ── Airflow (docker-compose.airflow.yml); start the MLflow stack first for the register/train tasks ──
AIRFLOW_COMPOSE ?= docker compose -f docker-compose.airflow.yml
.PHONY: airflow-up airflow-down airflow-logs airflow-check airflow-trigger

airflow-up:  ## Build and start Airflow on http://localhost:8080 (waits until healthy)
	$(AIRFLOW_COMPOSE) up -d --build --wait airflow

airflow-down:  ## Stop Airflow (keeps run history; `docker compose -f docker-compose.airflow.yml down -v` wipes it)
	$(AIRFLOW_COMPOSE) down

airflow-logs:  ## Follow Airflow's logs
	$(AIRFLOW_COMPOSE) logs -f --tail=100 airflow

airflow-check:  ## In the running container: DAG import errors (expect none) and the h1b_pipeline tasks
	$(AIRFLOW_COMPOSE) exec airflow airflow dags list-import-errors
	$(AIRFLOW_COMPOSE) exec airflow airflow tasks list h1b_pipeline

airflow-trigger:  ## Start one run of h1b_pipeline (unpauses it first); watch it at http://localhost:8080
	$(AIRFLOW_COMPOSE) exec airflow airflow dags unpause h1b_pipeline
	$(AIRFLOW_COMPOSE) exec airflow airflow dags trigger h1b_pipeline
