# Airflow image for the h1b pipeline DAG (dags/). Airflow and the project live in two separate
# Python environments inside one image:
#   - Airflow's own interpreter runs the scheduler, UI and DAG parsing, with Airflow's pinned libraries.
#   - /opt/h1b-venv runs the pipeline tasks (python -m h1b.<stage>) with the versions in requirements.txt.
# Keeping them apart means Airflow's pins (e.g. pydantic, sqlalchemy) never clash with mlflow/flaml/evidently.

# Pinned: same Airflow and Python versions on every machine. Python 3.11 matches CI and docker/Dockerfile.
FROM apache/airflow:3.3.2-python3.11

# LightGBM (used by FLAML models) needs the OpenMP runtime, which the slim Debian base lacks.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /opt/h1b mirrors the repo layout, so h1b.config resolves params.yaml, data/ and reports/ under it
# (data/ and reports/ are mounted at run time by docker compose, never baked into the image).
# /opt/h1b-venv is created here too: /opt is root-owned, so the airflow user can't create it later.
RUN mkdir -p /opt/h1b/data /opt/h1b/reports /opt/h1b-venv \
    && chown -R airflow:root /opt/h1b /opt/h1b-venv
USER airflow

RUN python -m venv /opt/h1b-venv

# Dependencies before code, so editing src/ doesn't reinstall them.
COPY --chown=airflow:root requirements.txt /opt/h1b/requirements.txt
RUN /opt/h1b-venv/bin/pip install --no-cache-dir -r /opt/h1b/requirements.txt

# Editable install on purpose (same as docker/Dockerfile): h1b.config finds the repo root from the
# package location, and MLflow needs `h1b` importable to unpickle our pyfunc models.
COPY --chown=airflow:root pyproject.toml params.yaml /opt/h1b/
COPY --chown=airflow:root src /opt/h1b/src
RUN /opt/h1b-venv/bin/pip install --no-cache-dir --no-deps -e /opt/h1b
