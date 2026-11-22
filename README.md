# H-1B Salary Prediction: an end-to-end MLOps pipeline

[![CI](https://github.com/alexbacro/h1b-salary-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/alexbacro/h1b-salary-mlops/actions/workflows/ci.yml)

Predict the **annualized offered salary** for an H-1B position from its job title, SOC
occupation code, employer, worksite location, and full-time status. The training data is the U.S. Department of
Labor's public LCA disclosure files. A job platform or candidate can use the prediction to judge whether an
offer is competitive.

**Team:** Jiwon Choi · Sola Shin · Alex Bacro · Faith Kirabo. See [docs/proposal.md](docs/proposal.md).

## Problem setup

| | |
|---|---|
| **Target** | `log(annual_wage)`: regression on log salary, since wages are right-skewed |
| **Features** | job title, SOC code, employer, worksite city/state, full-time flag |
| **Excluded** | prevailing-wage fields (`PW_*`), which would leak the target |
| **Metrics** | MAE ($), MAPE, log-RMSE |
| **Baseline** | median wage by (SOC code, state), falling back to the SOC median and then the global median |
| **Split** | temporal: train FY2022–23 · validate FY2024 · **test FY2025 held out** in `data/holdout/` |

## Architecture

```
            ┌──────── Airflow DAG (dags/h1b_pipeline.py): one task per stage, in this order ────────┐
DOL .xlsx ──► ingest ──► clean/validate ──► split ──┬─► baseline ──► register_baseline ──┐
 (data/raw)   (Parquet)   (wage normalization)      │                (only if no         │
                                                    │                 @champion yet)     ▼
                                                    └─► features ─► FLAML AutoML ─► MLflow tracking
                                                                      (train)       + Model Registry
                                                                                    │ @champion
    FY25 test data (clean + corrupted) ──► monitoring script ──► FastAPI /predict (Docker)
                                                   │                      │
                                                   └─► Evidently ◄── predictions
    (clean vs. corrupted FY25 batches; features rebuilt locally with prepare_request)
```

| Stage | Tooling | Status |
|---|---|---|
| Data ingestion, validation, baselines | pandas, Parquet | ✅ `src/h1b/` |
| Orchestration | Airflow DAG (Docker) | ✅ `dags/h1b_pipeline.py` |
| Experiment tracking & registry | MLflow + FLAML AutoML | ⏳ planned |
| Serving | FastAPI + Docker Compose | ✅ `src/h1b/serve.py` (`/health`, `/predict`, loads `@champion`); Docker via `docker-compose.yml` |
| Monitoring & drift simulation | Evidently | ✅ `make drift`, `make drift-sim`, `make api-check` |
| CI | GitHub Actions (ruff + pytest) | ✅ `.github/workflows/ci.yml` |

## Quickstart

Requires Python 3.11+.

```bash
git clone git@github.com:alexbacro/h1b-salary-mlops.git
cd h1b-salary-mlops
python3.11 -m venv .venv && source .venv/bin/activate
brew install libomp   # macOS only: xgboost/lightgbm need it (without it make test fails at collection)
make setup        # pip install -r requirements-dev.txt -e .
make test         # unit tests, no data needed
```

### Run everything, in order

The sections below explain each step. For the full pipeline end to end (Docker with Compose v2 required,
and the DOL files in `data/raw/`):

```bash
make data                                              # 1. ingest -> clean -> split -> baseline
make up                                                # 2. MLflow server (localhost:5001) + API (localhost:8000)
make register-docker                                   # 3. baseline -> @champion in the stack's MLflow
MLFLOW_TRACKING_URI=http://localhost:5001 make train   # 4. FLAML; promoted to @champion only if it beats the baseline
docker compose restart api && make smoke               # 5. API reloads and serves the new champion
make api-check                                         # 6. FY25 holdout (clean + corrupted) through the API
```

Or let Airflow run steps 1, 3 and 4: after `make up`, run `make airflow-up` and `make airflow-trigger`,
then continue with steps 5 and 6 (see [Run the pipeline with Airflow](#run-the-pipeline-with-airflow)).

Step 4 needs `MLFLOW_TRACKING_URI`: without it, `make train` registers into the local `mlflow.db`, which
the Docker API never reads (see the note on registries under [Run the API](#run-the-api)).

### Run the data stage

1. Download the LCA disclosure files listed in **[data/README.md](data/README.md)** into `data/raw/`
   (DOL blocks scripted downloads, so use a browser).
2. Run:

```bash
make data         # ingest -> clean -> split -> baseline
```

This writes:
- `data/processed/train.parquet`, `data/processed/val.parquet`, `data/holdout/test.parquet`
- `reports/ingest_report.json`: rows dropped by each validation rule, rows per fiscal year
- `reports/baseline_metrics_val.json`: the baseline MAE / MAPE / log-RMSE to beat

Fiscal years, filters and wage bounds are all set in [`params.yaml`](params.yaml).

### Run the API

The service loads whichever model version holds `h1b-salary-model@champion` in the MLflow registry
(name and alias under `serving:` in `params.yaml`), so register a model first:

```bash
make register-baseline   # registers the median baseline and sets @champion (needs `make data`)
make serve               # http://127.0.0.1:8000; interactive docs at /docs
```

- `GET /health` reports whether a model is loaded and which version is served.
- `POST /predict` takes `{"records": [...]}` (1 to 10,000 records with the columns in
  `REQUEST_FEATURES`, `src/h1b/features.py`) and returns one predicted annual wage in USD per record.
  Raw values such as `15-1252.00` or `ca` are normalized the same way as the training data, and
  `soc_major` is derived from `soc_code` (sending it is rejected).
- With no champion registered the service still starts, and `/predict` returns 503.
- The registry defaults to `mlflow.db` in the repo root; set `MLFLOW_TRACKING_URI` to use an MLflow server.
- **There are two registries, and they don't share models.** The local commands (`make register-baseline`,
  `make train`, `make serve`) use `mlflow.db` in the repo root. The Docker stack (`make up`,
  `make register-docker`, Airflow) uses the MLflow server, whose data lives in the `mlflow-data` volume.
  A model registered in one is invisible to the other, so an API started against the other one returns 503.
  To register into the Docker stack from your machine, prefix the command with
  `MLFLOW_TRACKING_URI=http://localhost:5001`.

### Run the stack with Docker

`docker-compose.yml` runs the MLflow tracking server (registry + artifact store) and the API that
serves `models:/h1b-salary-model@champion` from it. Requires Docker with Compose v2, and `make data`
for the registration step.

```bash
cp .env.example .env     # optional: change host ports (MLflow defaults to 5001, because macOS uses 5000)
make up                  # build the image, start mlflow + api, wait until both are healthy
make register-docker     # register the baseline in the stack's MLflow as @champion, then reload the api
make smoke               # GET /health and POST /predict
make logs                # follow the logs; `make down` stops the stack
```

- MLflow UI: http://localhost:5001. API docs: http://localhost:8000/docs.
- Both services use one image (`docker/Dockerfile`, the serving subset of `requirements.txt`).
- The registry (SQLite backend store and artifacts) lives in the `mlflow-data` volume, so it survives
  `make down`. `docker compose down -v` wipes it.
- The API loads the champion at startup. Until one is registered, `/health` shows `model_loaded: false`
  and `/predict` returns 503. After moving `@champion`, restart the API (`docker compose restart api`).
- The registration job mounts only `data/processed` (read-only). The holdout split is never mounted.

### Run the pipeline with Airflow

[`dags/h1b_pipeline.py`](dags/h1b_pipeline.py) runs the whole pipeline as one Airflow DAG, using the same
commands as the Makefile:

```
ingest → clean → split → baseline → register_baseline → train
```

Airflow runs in Docker (`docker-compose.airflow.yml`) and reads and writes the same `data/` and `reports/`
folders as `make data`. Requires Docker with Compose v2 and the DOL files in `data/raw/`.

```bash
make up                # MLflow server + API (docker-compose.yml); needed by the register and train tasks
make airflow-up        # build and start Airflow on http://localhost:8080
make airflow-trigger   # start one run; follow it in the UI (h1b_pipeline ▸ Grid)
make airflow-down      # stop Airflow
```

- `register_baseline` only runs when no model holds `@champion` yet, and is skipped otherwise.
- `train` promotes its FLAML model to `@champion` only if it beats the current champion's validation
  log-RMSE. Restart the API afterwards (`docker compose restart api`) so it serves the new version.
- Task details, testing and troubleshooting: [`dags/README.md`](dags/README.md).

## Repository layout

```
├── params.yaml              # single source of pipeline configuration
├── docker-compose.yml       # MLflow server + API (see "Run the stack with Docker")
├── docker/
│   ├── Dockerfile           # one image for MLflow server, API and the register job
│   ├── requirements-serve.txt  # the image's subset of requirements.txt
│   └── airflow.Dockerfile   # Airflow 3.3 + the project in its own venv (/opt/h1b-venv)
├── src/h1b/
│   ├── config.py            # loads params.yaml, resolves paths, default MLflow tracking URI
│   ├── schema.py            # canonical columns + mapping of year-to-year header variants
│   ├── ingest.py            # Excel -> Parquet (cached per file)
│   ├── clean.py             # dedupe, filter, annualize wages, normalize text, validate
│   ├── soc_crosswalk.py     # SOC 2010 -> 2018 lookup (generated)
│   ├── split.py             # temporal train/val/test split
│   ├── features.py          # feature contract: FEATURES, REQUEST_FEATURES, to_model_input, prepare_request
│   ├── baseline.py          # median-by-SOC-and-state baseline
│   ├── metrics.py           # MAE, MAPE, log-RMSE
│   ├── mlflow_models.py     # pyfunc wrappers: every registered model returns USD
│   ├── register_baseline.py # register the baseline in MLflow, set @champion
│   ├── encoding.py          # feature encoding for AutoML, fitted on train, shipped inside the model
│   ├── train.py             # FLAML AutoML -> MLflow; promote to @champion only if better
│   ├── serve.py             # FastAPI service: /health, /predict, loads @champion
│   ├── drift_report.py      # Evidently drift report, train vs val
│   ├── drift_simulation.py  # corrupt val and check the drift report flags it
│   └── api_check.py         # FY25 holdout through the API, drift + corruption checks
├── scripts/                 # one-off generators (e.g. build_soc_crosswalk.py)
├── dags/                    # Airflow DAG for the whole pipeline (see dags/README.md)
├── docker-compose.airflow.yml  # Airflow service (same network as docker-compose.yml)
├── tests/                   # pytest unit tests (run in CI)
├── data/                    # git-ignored except README; see data/README.md
├── reports/                 # generated JSON reports (git-ignored)
├── docs/                    # proposal, design notes, slide assets
└── .github/                 # CI workflow, PR template
```

## Contributing

- Branch from `main` (`feature/<short-name>`), open a PR, and let CI pass before merging.
- Run `make lint test` before pushing. `make format` auto-fixes most lint issues.
- Never commit data files, `mlruns/`, or credentials (`.gitignore` covers the usual ones).

## Monitoring

`make drift` and `make drift-sim` read `data/processed` (train = reference, val = current) and never touch
the holdout. `make api-check` is the one place the FY25 holdout is used, as the final production check.

- `make drift`: Evidently drift reports for the input features and, separately, the target (`log_wage`).
  Writes `reports/drift_features_train_vs_val.html`, `reports/drift_target_train_vs_val.html` and
  `reports/drift_train_vs_val.json`. The JSON has a top-level `alert` (true if any column is flagged or an
  unseen-share limit is exceeded) and the reasons.
- `make drift-sim`: corrupts a copy of val five ways (values that would pass input validation) and
  runs the same report on each. Writes `reports/drift_sim_<scenario>.html`,
  `reports/drift_simulation_summary.csv` and `reports/drift_simulation_summary.json`. It exits with
  code 1 if a scenario does not flag exactly its expected columns and raise exactly its expected
  unseen-share alerts.
- `make api-check`: needs the Docker stack running with a champion registered (`make up`,
  `make register-docker`, then `make train` for the trained model). It sends a random FY25 sample (`--n`,
  default 20,000, seed from `params.yaml`) to `POST /predict` in batches, clean and corrupted, rebuilds the
  model input locally with `prepare_request`, and runs the drift report against train. Clean FY25 is only
  reported; each corruption must flag its expected columns and raise its expected unseen-share alerts.
  Writes `reports/api_check_<scenario>.html` and `reports/api_check_summary.json` (model name and version
  from the API response, flags, unseen shares, MAE and MAPE against the real wages, prediction drift).
  Exit code 1 if a corruption is not caught.

**Decision rule.** A column is flagged when its drift score is at or above its Evidently threshold
(0.1 for Jensen-Shannon distance and normed Wasserstein distance). That is stricter than Evidently's
dataset-level `drift_share=0.5`. Medium- and high-cardinality columns are bucketed (top
`monitoring.top_n` train values, the rest as `other`), and the share of rows whose value never
appeared in train is checked against `monitoring.unseen_share_max` in [`params.yaml`](params.yaml).
The drift score alone misses moderate corruption: 10% new names flags neither name column, while the
unseen share of `job_title` goes from 0.156 (clean) to about 0.24. The limits are a first guess from
one validation year and should be revisited after the FY25 check.

**Results** (train FY22-23 vs val FY24, `main` with the SOC crosswalk and the O*NET overrides).
Clean val flags nothing (largest score 0.092, `job_title`). Each corruption flags exactly the columns it
corrupted:

| Scenario | Corruption | Flagged | Max score | Unseen-share alert |
|---|---|---|---|---|
| `clean` | none | - | 0.092 | - |
| `unseen_soc_30pct` | 30% of rows get SOC code `99-9999` | `soc_code`, `soc_major` | 0.343 | `soc_code` |
| `unseen_names_40pct` | 40% of rows get new employer and title | `employer_name`, `job_title` | 0.136 | `employer_name`, `job_title` |
| `full_time_shift_60pct` | 60% of rows set to part time | `full_time` | 0.497 | - |
| `swap_title_employer` | `job_title` and `employer_name` swapped | `employer_name`, `job_title` | 0.340 | `employer_name`, `job_title` |
| `wage_shift_plus_0.3` | `log_wage` + 0.3 (about 35% higher) | `log_wage` | 0.897 | - |

The drift report also found a real problem: before the SOC 2010 to 2018 crosswalk, `soc_code` scored 0.32
(flagged) between train and val; after it, 0.082, and 0.073 with the O*NET overrides.

**API-level check** (`make api-check`: FY25 holdout, random sample of 20,000 rows, seed 42, scored against train).
The input flags only depend on the data, so they are the same for both models. `job_title` is flagged in every
scenario that checks the inputs, including clean FY25 (0.109): that is natural drift from train, not a false alarm of the check.
`soc_code` on clean FY25 is close to the line (0.098). MAE is in dollars against the real wages. Prediction
drift compares the log of the predictions with predictions on a val sample.

| Scenario | Flagged | Unseen-share alert | MAE median baseline | MAE LightGBM | Prediction drift LightGBM |
|---|---|---|---|---|---|
| `clean_fy25` | `job_title` | - | 30,199 | 22,835 | 0.039 |
| `unseen_soc_30pct` | `job_title`, `soc_code`, `soc_major` | `soc_code` | 33,499 | 32,854 | 0.370 (flagged) |
| `unseen_names_40pct` | `employer_name`, `job_title` | `employer_name`, `job_title` | 30,199 | 25,448 | 0.035 |
| `full_time_shift_60pct` | `full_time`, `job_title` | - | 30,199 | 23,498 | 0.021 |
| `swap_title_employer` | `employer_name`, `job_title` | `employer_name`, `job_title` | 30,199 | 29,542 | 0.105 (flagged) |
| `wage_shift_plus_0.3` | `log_wage` (target, no API call) | - | - | - | - |

- The median baseline only reads `soc_code` and `worksite_state`, so only the SOC corruption changes its
  predictions. The LightGBM model (FLAML, 600 s budget, trained locally: validation MAE 20,412 against 27,648
  for the baseline) reacts to the other corruptions too. Prediction drift flags only the SOC corruption and the
  swap: the 40% unseen names raise MAE by 11% without a prediction-drift flag, so the input checks catch what
  the prediction check misses.
- Clean FY25 MAE (22,835) is higher than validation (20,412), possibly because wages keep rising; this was
  not tested on FY25.
- One sample and one seed, and the corruption sizes are our choice. FY25 is used only as this final production
  check, never for tuning. FLAML's search is time-budgeted, so a re-run gives slightly different numbers.

## Contributions

| Member | Responsibilities |
|---|---|
| Alex Bacro | Repository and CI setup, data ingestion pipeline, technical design, Docker compose stack (MLflow server, API), end-to-end smoke test in CI, PR review and merging |
| Jiwon Choi | MLflow experiment tracking, FLAML training, model registry (alias @champion) |
| Sola Shin | Evidently monitoring, drift simulation, project coordination |
| Faith Kirabo | Airflow orchestration: DAG for the whole pipeline (`dags/h1b_pipeline.py`), its Docker image and compose service, bootstrap-only baseline registration. FastAPI service (`src/h1b/serve.py`): `/health` and batch `/predict`, loads `h1b-salary-model@champion` by alias, input normalization shared with training (`features.prepare_request`), batch cap and clear 422/503 errors |
