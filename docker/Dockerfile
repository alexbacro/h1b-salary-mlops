# One image for the whole compose stack: the MLflow tracking server, the FastAPI service and the
# one-shot registration job all run from it (different commands in docker-compose.yml). One build,
# one set of pinned dependencies, and the API can unpickle models because `h1b` is installed.
FROM python:3.11-slim

# GIT_PYTHON_REFRESH=quiet: the image has no git, so MLflow's GitPython import would warn "Bad git executable".
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

# LightGBM/xgboost (used by FLAML models) need the OpenMP runtime, which the slim Debian base lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes don't reinstall them. requirements.txt is used only as a
# constraints file: the image gets the serving subset (no Evidently, no Excel reader), at the
# versions the rest of the repo pins.
# pip rejects extras in a constraints file ("Constraints cannot have extras"), and requirements.txt
# lists flaml[automl] once #13 lands, so the extras are stripped before using it with -c.
COPY requirements.txt docker/requirements-serve.txt ./
RUN sed -E 's/\[[^]]*\]//' requirements.txt > constraints.txt \
    && pip install -r requirements-serve.txt -c constraints.txt

# Editable install on purpose: h1b.config resolves params.yaml and data/ relative to the repo
# root (/app), and pickled models import h1b.* when MLflow loads them.
COPY pyproject.toml params.yaml ./
COPY src ./src
RUN pip install --no-deps -e .

# Unprivileged user. /mlflow is the server's state (named volume); data/ and reports/ must be
# writable because h1b.config.get_paths creates the data directories on startup.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /mlflow data reports \
    && chown -R app /mlflow data reports
USER app

EXPOSE 8000 5000
CMD ["uvicorn", "h1b.serve:app", "--host", "0.0.0.0", "--port", "8000"]
