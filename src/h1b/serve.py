"""Serving stage: FastAPI service that predicts the annual offered wage for H-1B positions.

The model is loaded from the MLflow Model Registry by alias (models:/<name>@<alias>, set in
params.yaml under `serving`), never from a local file. Promoting a new model therefore only
means moving the alias in MLflow and restarting the service; no API code changes.

If no model holds the alias yet (or MLflow is unreachable), the service still starts:
/health reports model_loaded=false and /predict returns 503 until a model is registered.

Usage:
    make serve            # uvicorn h1b.serve:app on http://127.0.0.1:8000 (docs at /docs)
    MLFLOW_TRACKING_URI=http://localhost:5000 make serve   # use an MLflow server instead of <repo>/mlflow.db
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from h1b.config import load_params
from h1b.features import validate_features
from h1b.register_baseline import DEFAULT_TRACKING_URI

log = logging.getLogger(__name__)

MODEL_OUTPUTS = ("log_wage", "dollars")


# ── Request / response schemas ────────────────────────────────────────────────


class JobRecord(BaseModel):
    """One position, using the model's feature columns from h1b.features.

    Fields must match features.FEATURES (a test enforces this). Values are expected in the
    same normalized form as data/processed/*.parquet, e.g. upper-case job titles.
    """

    # Unknown fields are rejected so a misspelled column fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    soc_major: str = Field(examples=["15"])
    naics_2: str | None = Field(default=None, examples=["54"])  # NAICS is optional in the DOL files
    worksite_state: str = Field(examples=["IL"])
    soc_code: str = Field(examples=["15-2051"])
    worksite_city: str | None = Field(default=None, examples=["CHICAGO"])
    job_title: str = Field(examples=["DATA SCIENTIST"])
    employer_name: str = Field(examples=["ACME CORP"])
    full_time: bool = Field(examples=[True])


class PredictRequest(BaseModel):
    # A list so the FY2025 holdout can be scored in batches; a single position is a list of one.
    records: list[JobRecord] = Field(min_length=1)


class PredictResponse(BaseModel):
    predictions: list[float]  # annual wage in dollars, one per record, same order
    model_name: str
    model_alias: str
    model_version: str


# ── Model loading ─────────────────────────────────────────────────────────────


def load_champion(model_name: str, alias: str) -> tuple[Any, str]:
    """Resolve alias -> version, then load that exact version.

    Loading by the resolved version (not by alias directly) guarantees the reported
    version is the one actually served, even if the alias moves during startup.
    """
    version = MlflowClient().get_model_version_by_alias(model_name, alias).version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    return model, str(version)


def create_app(serving: dict[str, Any] | None = None) -> FastAPI:
    """Build the app. `serving` defaults to params.yaml; tests pass their own."""
    cfg = serving or load_params()["serving"]
    if cfg["model_output"] not in MODEL_OUTPUTS:
        raise ValueError(f"serving.model_output must be one of {MODEL_OUTPUTS}, got {cfg['model_output']!r}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Docker sets MLFLOW_TRACKING_URI; locally fall back to the repo's mlflow.db (same default as training).
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
        app.state.model, app.state.model_version = None, None
        try:
            app.state.model, app.state.model_version = load_champion(cfg["model_name"], cfg["model_alias"])
            log.info("loaded %s@%s (version %s)", cfg["model_name"], cfg["model_alias"], app.state.model_version)
        # Broad on purpose: missing alias, unreachable server, or a bad artifact should all
        # leave the API running (but not ready) rather than crash the container on startup.
        except Exception as exc:
            log.warning("no model loaded for %s@%s: %s", cfg["model_name"], cfg["model_alias"], exc)
        yield

    app = FastAPI(title="H-1B salary prediction", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": request.app.state.model is not None,
            "model_name": cfg["model_name"],
            "model_alias": cfg["model_alias"],
            "model_version": request.app.state.model_version,
        }

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        model = request.app.state.model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"No model loaded for {cfg['model_name']}@{cfg['model_alias']}. "
                "Register one in MLflow and restart the service.",
            )
        # Same column order and dtypes the model was trained on.
        features = validate_features(pd.DataFrame([r.model_dump() for r in body.records]))
        raw = np.asarray(model.predict(features), dtype=float)
        wages = np.exp(raw) if cfg["model_output"] == "log_wage" else raw
        return PredictResponse(
            predictions=[round(float(w), 2) for w in wages],
            model_name=cfg["model_name"],
            model_alias=cfg["model_alias"],
            model_version=request.app.state.model_version,
        )

    return app


app = create_app()
