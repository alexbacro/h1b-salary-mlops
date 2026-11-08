"""Train the median baseline and register it in the MLflow Model Registry.

Registers the model named in params.yaml (`serving.model_name`), tags the new version with its
semantic version and moves the alias (`serving.model_alias`) to it, but only after a self-check
that reloads that version and reproduces its logged metrics. Consumers load `models:/<name>@<alias>`.
The model takes the feature columns in `h1b.features` and returns annual wage in USD.

Usage:
    python -m h1b.register_baseline
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m h1b.register_baseline
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from h1b.baseline import MedianBaseline
from h1b.config import DEFAULT_TRACKING_URI, REPO_ROOT, get_paths, load_params
from h1b.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURES, to_model_input
from h1b.metrics import regression_report
from h1b.mlflow_models import BaselinePyfunc

log = logging.getLogger(__name__)

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "mlruns"
EXPERIMENT_NAME = "h1b-salary"
INITIAL_SEMVER = "0.1.0"  # first version; later ones bump the patch number (see _next_semver)

SIGNATURE = ModelSignature(
    inputs=Schema(
        [ColSpec("string", c) for c in CATEGORICAL_FEATURES]
        # Booleans are sent as float64 (1.0 / 0.0 / NaN): MLflow's boolean ColSpec cannot hold nulls.
        + [ColSpec("double", c, required=False) for c in BOOLEAN_FEATURES]
    ),
    outputs=Schema([ColSpec("double")]),
)


def _next_semver(client: MlflowClient, model_name: str) -> str:
    """INITIAL_SEMVER for the first tagged version, else the highest existing `semver` tag with its patch + 1.

    MLflow numbers versions 1, 2, 3...; the semantic version is our own tag on top of that.
    """
    found = []
    for mv in client.search_model_versions(f"name='{model_name}'"):
        tag = mv.tags.get("semver")
        if tag and re.fullmatch(r"\d+\.\d+\.\d+", tag):
            found.append(tuple(int(part) for part in tag.split(".")))
    if not found:
        return INITIAL_SEMVER
    major, minor, patch = max(found)
    return f"{major}.{minor}.{patch + 1}"


def register(
    train: pd.DataFrame,
    val: pd.DataFrame,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    params: dict[str, Any] | None = None,
) -> str:
    """Fit, log and register the baseline; promote it only if the self-check passes. Returns the version."""
    params = params or load_params()
    model_name = params["serving"]["model_name"]
    alias = params["serving"]["model_alias"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        location = (Path(artifact_root) / EXPERIMENT_NAME).resolve().as_uri()
        exp_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=location)
    else:
        exp_id = exp.experiment_id

    model = MedianBaseline().fit(train)
    metrics = regression_report(val["annual_wage"], model.predict(val))

    split = params.get("split", {})
    with mlflow.start_run(experiment_id=exp_id, run_name="median-baseline"):
        mlflow.log_params(
            {
                "model_type": "median_by_soc_state",
                "features": ",".join(FEATURES),
                "train_years": str(split.get("train_years")),
                "val_years": str(split.get("val_years")),
                "seed": params.get("seed"),
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=BaselinePyfunc(model),
            signature=SIGNATURE,
            registered_model_name=model_name,
        )

    version = getattr(info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        version = max(int(v.version) for v in versions)
    version = str(version)

    # Self-check BEFORE promotion: reload this exact version and reproduce the logged metrics.
    # If it fails, the alias still points at the previous model (or nothing).
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    check = regression_report(val["annual_wage"], loaded.predict(to_model_input(val)))
    for name, value in metrics.items():
        if not np.isclose(check[name], value):
            raise RuntimeError(f"Registered model disagrees on {name}: {check[name]} vs {value}")

    semver = _next_semver(client, model_name)
    client.set_model_version_tag(model_name, version, "semver", semver)
    client.set_registered_model_alias(model_name, alias, version)
    log.info("registered %s v%s (semver %s) as @%s: %s", model_name, version, semver, alias, metrics)
    return version


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    paths = get_paths(params)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    train = pd.read_parquet(paths.processed / "train.parquet")
    val = pd.read_parquet(paths.processed / "val.parquet")  # the isolated test split is never touched here
    register(train, val, tracking_uri=tracking_uri, params=params)
    log.info("tracking URI: %s", tracking_uri)
    log.info("model URI:    models:/%s@%s", params["serving"]["model_name"], params["serving"]["model_alias"])


if __name__ == "__main__":
    main()
