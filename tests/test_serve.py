"""API tests against a throwaway local MLflow registry (SQLite in tmp_path); no server or real model needed."""

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from h1b.features import FEATURES
from h1b.serve import MAX_BATCH_SIZE, JobRecord, create_app

MODEL_NAME = "test-salary-model"
CONSTANT_WAGE = 100_000.0

RECORD = {
    "soc_major": "15",
    "naics_2": "54",
    "worksite_state": "IL",
    "soc_code": "15-2051",
    "worksite_city": "CHICAGO",
    "job_title": "DATA SCIENTIST",
    "employer_name": "ACME CORP",
    "full_time": True,
}


class ConstantModel(mlflow.pyfunc.PythonModel):
    """Stands in for the real model: predicts $100,000 for every row (registered models return USD, #8)."""

    def predict(self, context, model_input, params=None):
        return [CONSTANT_WAGE] * len(model_input)


def serving_cfg():
    return {"model_name": MODEL_NAME, "model_alias": "champion"}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Point MLflow at an empty SQLite registry; artifacts stay inside tmp_path."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    yield experiment_id
    mlflow.set_tracking_uri(None)


def register_champion(experiment_id, signature=None) -> str:
    """Register ConstantModel (optionally with an input signature) and give it the @champion alias."""
    with mlflow.start_run(experiment_id=experiment_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ConstantModel(),
            registered_model_name=MODEL_NAME,
            signature=signature,
            pip_requirements=["mlflow"],  # skip dependency inference to keep the test fast
        )
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", info.registered_model_version)
    return str(info.registered_model_version)


@pytest.fixture
def champion(registry):
    return register_champion(registry)


@pytest.fixture
def strict_champion(registry):
    """A champion whose signature requires a column the API never sends, so MLflow's schema check fails."""
    columns = [ColSpec("string", c) for c in FEATURES] + [ColSpec("string", "not_a_feature")]
    return register_champion(registry, signature=ModelSignature(inputs=Schema(columns)))


def test_request_fields_match_feature_contract():
    assert list(JobRecord.model_fields) == FEATURES


def test_no_champion_starts_but_is_not_ready(registry):
    with TestClient(create_app(serving_cfg())) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_loaded"] is False
        assert client.post("/predict", json={"records": [RECORD]}).status_code == 503


def test_predict_with_champion_returns_dollars(champion):
    with TestClient(create_app(serving_cfg())) as client:
        assert client.get("/health").json()["model_version"] == champion
        resp = client.post("/predict", json={"records": [RECORD, RECORD]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"] == [pytest.approx(CONSTANT_WAGE)] * 2  # passed through unchanged
    assert body["model_version"] == champion


@pytest.mark.parametrize(
    "body",
    [
        {"records": []},  # empty batch
        {"records": [RECORD] * (MAX_BATCH_SIZE + 1)},  # batch over the cap
        {"records": [{k: v for k, v in RECORD.items() if k != "soc_code"}]},  # missing feature
        {"records": [{**RECORD, "salary": 1}]},  # unknown field
    ],
)
def test_bad_input_is_rejected(champion, body):
    with TestClient(create_app(serving_cfg())) as client:
        assert client.post("/predict", json=body).status_code == 422


def test_model_schema_mismatch_returns_422_with_reason(strict_champion):
    with TestClient(create_app(serving_cfg())) as client:
        resp = client.post("/predict", json={"records": [RECORD]})
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("Model rejected the input:")
    assert "not_a_feature" in resp.json()["detail"]  # MLflow's message names the problem column
