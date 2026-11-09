# Convenience targets. Run `make help` to list them.
PYTHON ?= python
COMPOSE ?= docker compose
-include .env   # optional overrides shared with docker compose (see .env.example)
API_PORT ?= 8000

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

# ── Docker compose stack (MLflow server + API); see "Run the stack with Docker" in the README ──
.PHONY: up down logs register-docker smoke

up:  ## Build and start MLflow (localhost:5001) + API (localhost:8000) in Docker
	$(COMPOSE) up -d --build --wait mlflow api

down:  ## Stop the stack (keeps the registry volume; add -v by hand to wipe it)
	$(COMPOSE) down

logs:  ## Follow the stack's logs
	$(COMPOSE) logs -f --tail=100

register-docker:  ## Register the baseline in the stack's MLflow (needs make data), then reload the API
	$(COMPOSE) run --rm register
	$(COMPOSE) restart api
	$(COMPOSE) up -d --wait api

smoke:  ## GET /health and POST /predict against the running API
	curl -fsS http://localhost:$(API_PORT)/health; echo
	curl -fsS -X POST http://localhost:$(API_PORT)/predict -H 'Content-Type: application/json' \
	  -d '{"records": [{"soc_code": "15-1252", "worksite_state": "CA", "worksite_city": "San Jose", "job_title": "Software Engineer", "employer_name": "Acme Corp", "naics_2": "54", "full_time": null}]}'; echo
