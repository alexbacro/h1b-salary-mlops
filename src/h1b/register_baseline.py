"""Train the median baseline and register it in the MLflow Model Registry.

Registers `h1b-salary-model`, tags the new version with its semantic version and
points the `champion` alias at it. Consumers load `models:/h1b-salary-model@champion`.
The model takes the feature columns in `h1b.features` and returns annual wage in USD.

Usage:
    python -m h1b.register_baseline
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m h1b.register_baseline
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from h1b.baseline import MedianBaseline
from h1b.config import REPO_ROOT, get_paths, load_params
from h1b.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURES, validate_features
from h1b.metrics import regression_report
from h1b.mlflow_models import BaselinePyfunc

log = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "mlruns"
EXPERIMENT_NAME = "h1b-salary"
REGISTERED_MODEL_NAME = "h1b-salary-model"
CHAMPION_ALIAS = "champion"
SEMVER = "0.1.0"  # MLflow auto-numbers versions; the semantic version is stored as a tag

SIGNATURE = ModelSignature(
    inputs=Schema(
        [ColSpec("string", c) for c in CATEGORICAL_FEATURES]
        + [ColSpec("boolean", c, required=False) for c in BOOLEAN_FEATURES]
    ),
    outputs=Schema([ColSpec("double")]),
)


def register(
    train: pd.DataFrame,
    val: pd.DataFrame,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    params: dict[str, Any] | None = None,
) -> str:
    """Fit, log, register and promote the baseline. Returns the registry version."""
    params = params or {}
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
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    version = getattr(info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        version = max(int(v.version) for v in versions)
    version = str(version)

    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "semver", SEMVER)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)

    # Smoke test: load through the alias and reproduce the logged metrics.
    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    check = regression_report(val["annual_wage"], loaded.predict(validate_features(val)))
    for name, value in metrics.items():
        if not np.isclose(check[name], value):
            raise RuntimeError(f"Registered model disagrees on {name}: {check[name]} vs {value}")
    log.info("registered %s v%s (semver %s): %s", REGISTERED_MODEL_NAME, version, SEMVER, metrics)
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
    log.info("model URI:    models:/%s@%s", REGISTERED_MODEL_NAME, CHAMPION_ALIAS)


if __name__ == "__main__":
    main()
