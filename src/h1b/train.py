"""Train a FLAML AutoML model on log_wage, log it to MLflow and promote it to @champion if it wins.

The model is registered under `serving.model_name` (params.yaml) with its preprocessing inside, so the
API keeps sending the raw `h1b.features` columns and gets annual wage in USD back (#8). Promotion happens
only after a self-check that reloads the version and reproduces its metrics and, by default, only if its
validation log-RMSE beats the current champion's.

Usage:
    python -m h1b.train
    python -m h1b.train --time-budget 120 --max-train-rows 200000   # quick smoke run
    python -m h1b.train --promote never                             # register, keep the champion
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from flaml import AutoML
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from h1b.config import DEFAULT_TRACKING_URI, get_paths, load_params
from h1b.encoding import FeatureEncoder
from h1b.features import FEATURES, TARGET, to_model_input
from h1b.metrics import regression_report
from h1b.mlflow_models import FlamlPyfunc
from h1b.register_baseline import EXPERIMENT_NAME, SIGNATURE, _next_semver

log = logging.getLogger(__name__)


def champion_log_rmse(client: MlflowClient, model_name: str, alias: str) -> tuple[bool, float | None]:
    """(has_champion, its validation log-RMSE), read from the `val_log_rmse` tag or, failing that, its run."""
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return False, None
    tag = mv.tags.get("val_log_rmse")
    if tag is not None:
        return True, float(tag)
    if mv.run_id:
        value = client.get_run(mv.run_id).data.metrics.get("val_log_rmse")
        if value is not None:
            return True, float(value)
    return True, None


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict[str, Any],
    tracking_uri: str = DEFAULT_TRACKING_URI,
    promote: str = "if-better",
) -> dict[str, Any]:
    """Fit, log, register and (maybe) promote. `promote`: "if-better" | "always" | "never"."""
    cfg = params["training"]
    model_name = params["serving"]["model_name"]
    alias = params["serving"]["model_alias"]
    seed = params.get("seed", 42)
    if cfg.get("max_train_rows"):
        train_df = train_df.sample(n=min(int(cfg["max_train_rows"]), len(train_df)), random_state=seed)

    encoder = FeatureEncoder().fit(train_df)
    x_train, y_train = encoder.transform(train_df), train_df[TARGET]
    x_val, y_val = encoder.transform(val_df), val_df[TARGET]

    # No local artifact_location here: against an MLflow server it would put artifacts on this laptop.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name="flaml-automl"):
        mlflow.log_params(
            {
                "model_type": "flaml_automl",
                "features": ",".join(FEATURES),
                "estimator_list": ",".join(cfg["estimator_list"]),
                "time_budget_s": cfg["time_budget_s"],
                "metric": cfg["metric"],
                "train_rows": len(train_df),
                "seed": seed,
            }
        )
        automl = AutoML()
        automl.fit(
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            task="regression",
            metric=cfg["metric"],
            estimator_list=cfg["estimator_list"],
            time_budget=cfg["time_budget_s"],
            eval_method="holdout",
            seed=seed,
            mlflow_logging=True,
        )
        metrics = regression_report(val_df["annual_wage"], np.exp(automl.predict(x_val)))
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_dict(
            {k: (v.item() if hasattr(v, "item") else v) for k, v in automl.best_config.items()},
            "best_config.json",
        )
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FlamlPyfunc(encoder, automl),
            signature=SIGNATURE,
            registered_model_name=model_name,
        )

    version = getattr(info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        version = max(int(v.version) for v in versions)
    version = str(version)

    # Self-check BEFORE promotion: reload this exact version and reproduce the logged metrics.
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    check = regression_report(val_df["annual_wage"], loaded.predict(to_model_input(val_df)))
    for key, value in metrics.items():
        if not np.isclose(check[key], value, rtol=1e-6):
            raise RuntimeError(f"Registered model disagrees on {key}: {check[key]} vs {value}")

    semver = _next_semver(client, model_name)
    client.set_model_version_tag(model_name, version, "semver", semver)
    client.set_model_version_tag(model_name, version, "val_log_rmse", str(metrics["log_rmse"]))
    client.set_model_version_tag(model_name, version, "estimator", str(automl.best_estimator))

    has_champion, champion = champion_log_rmse(client, model_name, alias)
    if promote == "always":
        promoted = True
    elif promote == "never":
        promoted = False
    elif not has_champion:
        promoted = True
    elif champion is None:
        log.warning("Champion has no val_log_rmse to compare against; not promoting (use --promote always)")
        promoted = False
    else:
        promoted = metrics["log_rmse"] < champion
    if promoted:
        client.set_registered_model_alias(model_name, alias, version)

    log.info(
        "registered %s v%s (semver %s, %s): %s | champion log-RMSE %s | promoted: %s",
        model_name, version, semver, automl.best_estimator, metrics, champion, promoted,
    )  # fmt: skip
    return {
        "version": version,
        "semver": semver,
        "metrics": metrics,
        "promoted": promoted,
        "champion_log_rmse": champion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time-budget", type=float, help="override training.time_budget_s")
    parser.add_argument("--max-train-rows", type=int, help="override training.max_train_rows")
    parser.add_argument("--promote", choices=["if-better", "always", "never"], default="if-better")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    params = load_params()
    params["training"] = dict(params["training"])
    if args.time_budget is not None:
        params["training"]["time_budget_s"] = args.time_budget
    if args.max_train_rows is not None:
        params["training"]["max_train_rows"] = args.max_train_rows

    paths = get_paths(params)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    train_df = pd.read_parquet(paths.processed / "train.parquet")
    val_df = pd.read_parquet(paths.processed / "val.parquet")  # the isolated test split is never touched here
    train(train_df, val_df, params, tracking_uri=tracking_uri, promote=args.promote)
    log.info("tracking URI: %s", tracking_uri)


if __name__ == "__main__":
    main()
