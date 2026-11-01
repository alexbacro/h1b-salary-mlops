"""API tests against a throwaway local MLflow registry (SQLite in tmp_path); no server or real model needed."""

import math

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from h1b.features import FEATURES
from h1b.serve import JobRecord, create_app

MODEL_NAME = "test-salary-model"
CONSTANT_LOG_WAGE = math.log(100_000)

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
    """Stands in for the real model: predicts log($100,000) for every row."""

    def predict(self, context, model_input, params=None):
        return [CONSTANT_LOG_WAGE] * len(model_input)


def serving_cfg(model_output="log_wage"):
    return {"model_name": MODEL_NAME, "model_alias": "champion", "model_output": model_output}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Point MLflow at an empty SQLite registry; artifacts stay inside tmp_path."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    yield experiment_id
    mlflow.set_tracking_uri(None)


@pytest.fixture
def champion(registry):
    """Register ConstantModel and give it the @champion alias."""
    with mlflow.start_run(experiment_id=registry):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ConstantModel(),
            registered_model_name=MODEL_NAME,
            pip_requirements=["mlflow"],  # skip dependency inference to keep the test fast
        )
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", info.registered_model_version)
    return str(info.registered_model_version)


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
    assert body["predictions"] == [pytest.approx(100_000, abs=0.01)] * 2  # exp(log wage) -> dollars
    assert body["model_version"] == champion


def test_dollars_output_is_not_exponentiated(champion):
    with TestClient(create_app(serving_cfg(model_output="dollars"))) as client:
        resp = client.post("/predict", json={"records": [RECORD]})
    assert resp.json()["predictions"] == [pytest.approx(CONSTANT_LOG_WAGE, abs=0.01)]


@pytest.mark.parametrize(
    "body",
    [
        {"records": []},  # empty batch
        {"records": [{k: v for k, v in RECORD.items() if k != "soc_code"}]},  # missing feature
        {"records": [{**RECORD, "salary": 1}]},  # unknown field
    ],
)
def test_bad_input_is_rejected(champion, body):
    with TestClient(create_app(serving_cfg())) as client:
        assert client.post("/predict", json=body).status_code == 422


def test_invalid_model_output_setting_fails_fast():
    with pytest.raises(ValueError, match="model_output"):
        create_app(serving_cfg(model_output="euros"))
