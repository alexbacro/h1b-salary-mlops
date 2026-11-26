# Convenience targets. Run `make help` to list them.
PYTHON ?= python
ARGS ?=

COMPOSE ?= docker compose
-include .env   # optional overrides shared with docker compose (see .env.example)
API_PORT ?= 8000

.PHONY: help setup lint format test ingest clean-data split baseline register-baseline train data drift drift-sim serve

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

train:  ## FLAML AutoML on log_wage -> MLflow; promotes @champion only if it wins (e.g. ARGS="--time-budget 120")
	$(PYTHON) -m h1b.train $(ARGS)
drift-sim:  ## Corrupt val in several ways and check Evidently flags it -> reports/drift_sim_*.html
	$(PYTHON) -m h1b.drift_simulation

# Bound to 127.0.0.1 so the dev server is not exposed on the network; Docker sets its own host.
serve:  ## FastAPI service on http://127.0.0.1:8000 (registry: MLFLOW_TRACKING_URI, else ./mlflow.db)
	$(PYTHON) -m uvicorn h1b.serve:app --host 127.0.0.1 --port 8000

# ── Docker compose stack (MLflow server + API); see "Run the stack with Docker" in the README ──
.PHONY: up down logs register-docker smoke docker-smoke

# mlflow and api share one image tag. Build it once (via api) before starting: with `up --build`
# both services build and race to export the same tag, and a fresh machine fails with "already exists".
up:  ## Build and start MLflow (localhost:5001) + API (localhost:8000) in Docker
	$(COMPOSE) build api
	$(COMPOSE) up -d --wait mlflow api

down:  ## Stop the stack (keeps the registry volume; add -v by hand to wipe it)
	$(COMPOSE) down

logs:  ## Follow the stack's logs
	$(COMPOSE) logs -f --tail=100

register-docker:  ## Register the baseline in the stack's MLflow (needs make data), then reload the API
	$(COMPOSE) run --rm register
	$(COMPOSE) restart api
	$(COMPOSE) up -d --wait api

docker-smoke:  ## CI's end-to-end Docker test on synthetic data (own compose project; cleans up after itself)
	PYTHON=$(PYTHON) scripts/docker_smoke.sh

smoke:  ## GET /health and POST /predict against the running API
	curl -fsS http://localhost:$(API_PORT)/health; echo
	curl -fsS -X POST http://localhost:$(API_PORT)/predict -H 'Content-Type: application/json' \
	  -d '{"records": [{"soc_code": "15-1252", "worksite_state": "CA", "worksite_city": "San Jose", "job_title": "Software Engineer", "employer_name": "Acme Corp", "naics_2": "54", "full_time": null}]}'; echo

.PHONY: api-check
api-check:  ## FY25 holdout through the API in Docker, then drift + corruption checks -> reports/api_check_*
	$(PYTHON) -m h1b.api_check --url http://localhost:$(API_PORT)

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
