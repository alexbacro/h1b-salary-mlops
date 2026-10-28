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
DOL .xlsx ──► ingest ──► clean/validate ──► split ──┬─► baseline
 (data/raw)   (Parquet)   (wage normalization)       │
                                                    └─► features ─► FLAML AutoML ─► MLflow tracking
                                                                                    + Model Registry
                                                                                        │
                    Evidently dashboard ◄── prediction logs ◄── FastAPI /predict (Docker)
                    (clean vs. drifted test data)
```

| Stage | Tooling | Status |
|---|---|---|
| Data ingestion, validation, baselines | pandas, Parquet | ✅ `src/h1b/` |
| Orchestration | Airflow DAG | ⏳ planned |
| Experiment tracking & registry | MLflow + FLAML AutoML | ⏳ planned |
| Serving | FastAPI + Docker Compose | ⏳ planned |
| Monitoring & drift simulation | Evidently | ⏳ planned |
| CI | GitHub Actions (ruff + pytest) | ✅ `.github/workflows/ci.yml` |

## Quickstart

Requires Python 3.11+.

```bash
git clone git@github.com:alexbacro/h1b-salary-mlops.git
cd h1b-salary-mlops
python3.11 -m venv .venv && source .venv/bin/activate
make setup        # pip install -r requirements-dev.txt -e .
make test         # unit tests, no data needed
```

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

## Repository layout

```
├── params.yaml              # single source of pipeline configuration
├── src/h1b/
│   ├── config.py            # loads params.yaml, resolves paths
│   ├── schema.py            # canonical columns + mapping of year-to-year header variants
│   ├── ingest.py            # Excel -> Parquet (cached per file)
│   ├── clean.py             # dedupe, filter, annualize wages, normalize text, validate
│   ├── split.py             # temporal train/val/test split
│   ├── baseline.py          # median-by-SOC-and-state baseline
│   └── metrics.py           # MAE, MAPE, log-RMSE
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

## Contributions

| Member | Responsibilities |
|---|---|
| Alex Bacro | Repository & CI setup, data ingestion pipeline, technical design |
| Jiwon Choi | _TBD_ |
| Sola Shin | _TBD_ |
| Faith Kirabo | _TBD_ |
