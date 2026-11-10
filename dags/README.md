# Airflow DAG

`h1b_pipeline.py` orchestrates the whole pipeline, from the DOL files to a model promoted to `@champion`:

```
ingest → clean → split → baseline → register_baseline → train
```

Each task runs the same command as the Makefile (`python -m h1b.<stage>`), so `make data` and this DAG
always do the same thing. The DAG holds no pipeline logic of its own; settings come from `params.yaml`.

## Tasks

| Task | Command | Reads → writes |
|---|---|---|
| `ingest` | `python -m h1b.ingest` | `data/raw/*.xlsx` → `data/interim/*.parquet` (cached per file) |
| `clean` | `python -m h1b.clean` | `data/interim/` → `data/processed/lca_clean.parquet`, `reports/ingest_report.json` |
| `split` | `python -m h1b.split` | → `data/processed/{train,val}.parquet`, `data/holdout/test.parquet` |
| `baseline` | `python -m h1b.baseline` | → `reports/baseline_metrics_val.json` |
| `register_baseline` | `python -m h1b.register_baseline` | registers the median baseline in MLflow, **only if no `@champion` exists yet** |
| `train` | `python -m h1b.train --promote if-better` | trains FLAML, registers it, moves `@champion` only if its validation log-RMSE is lower |

- **`register_baseline` is a bootstrap.** `register_baseline` always moves the alias, so running it on every
  run would demote a better FLAML champion. A check runs first: if a champion exists, the task exits with
  code 99 and shows as **skipped**. If MLflow can't be reached, the task **fails** rather than skipping.
- **`train` runs whether `register_baseline` succeeded or was skipped** (`trigger_rule="none_failed"`), and
  stops if it failed.
- The DAG is triggered manually (`schedule=None`, no backfill): DOL publishes new files quarterly, and they
  are downloaded by hand. Each task retries once after 2 minutes.

## How the tasks run

Airflow runs in Docker (`docker-compose.airflow.yml`, image `docker/airflow.Dockerfile`, Airflow 3.3.2 on
Python 3.11). The image has **two Python environments**:

- Airflow's own interpreter runs the scheduler, the UI and DAG parsing, with Airflow's pinned libraries.
- `/opt/h1b-venv` runs the tasks, with the versions in `requirements.txt` and `h1b` installed from `/opt/h1b`.

Airflow pins many libraries (pydantic, sqlalchemy, ...) that can clash with mlflow, flaml and evidently;
keeping the environments apart avoids that.

`data/` and `reports/` are mounted from the repo, so the DAG reads and writes the same files as `make data`
(`data/raw` is read-only). `dags/` is mounted too, so DAG edits are picked up without rebuilding the image.
The register and train tasks reach the MLflow server at `http://mlflow:5000`; the compose file uses the same
project name as `docker-compose.yml`, so both stacks share one network.

## Run it

Requires Docker with Compose v2, and the DOL files in `data/raw/` (see `data/README.md`). The register and
train tasks also need the MLflow stack from `docker-compose.yml` (#10) and `h1b.train` (#13); without them,
the four data tasks still run.

```bash
make up                # MLflow server + API (docker-compose.yml); needed by register_baseline and train
make airflow-up        # build and start Airflow on http://localhost:8080 (local only, no login)
make airflow-check     # expect no import errors and the six task names
make airflow-trigger   # start one run; follow it in the UI: h1b_pipeline ▸ Grid
make airflow-logs      # follow Airflow's logs
make airflow-down      # stop Airflow (run history is kept in the airflow-home volume)
```

In the Grid view each task is a box: green = success, pink = skipped, red = failed. Click a box ▸ Logs to see
its output (e.g. the rows kept by `clean`, or the baseline's MAE).

**After `train` promotes a new champion, restart the API** (`docker compose restart api`). The API loads the
champion at startup and keeps serving the previous version until it restarts.

## Test it

- `tests/test_dag.py` checks that the DAG loads without import errors, the task order, the manual schedule,
  the bootstrap exit code and `train`'s trigger rule. Airflow is not in `requirements-dev.txt`, so CI
  **skips** these tests; run `pytest tests/test_dag.py` in an environment that has Airflow 3.3 installed.
- `make airflow-check` runs the same import check inside the running container.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `register_baseline` fails with a connection error | The MLflow stack isn't running: `make up` first |
| `train` fails with `No module named h1b.train` | Training (#13) isn't in this branch yet |
| `train` is killed with exit code 137 | Docker ran out of memory: raise it in Docker Desktop ▸ Settings ▸ Resources, or set `training.max_train_rows` in `params.yaml` (added by #13) |
| The API still serves the old version | Restart it after promotion: `docker compose restart api` |
| A DAG edit doesn't show in the UI | Airflow re-parses `dags/` periodically; wait about a minute or run `make airflow-check` |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AIRFLOW_PORT` | `8080` | Host port of the Airflow UI |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Registry used by `register_baseline` and `train` |
| `H1B_PYTHON` / `H1B_ROOT` | `/opt/h1b-venv/bin/python` / `/opt/h1b` | Interpreter and project root the tasks use (override to run the DAG outside the image) |

## Follow-ups

- Move the `airflow` service into `docker-compose.yml` once #10 is in `main`, so one `make up` starts the
  whole stack.
- Reload the API automatically after `train` promotes a new champion.
