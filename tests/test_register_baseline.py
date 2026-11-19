"""The baseline registers, gets its semver tag and loads through the alias; failures never promote."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from h1b import register_baseline
from h1b.config import load_params
from h1b.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, DTYPES, to_model_input
from h1b.register_baseline import INITIAL_SEMVER, register

PARAMS = load_params()
MODEL_NAME = PARAMS["serving"]["model_name"]
ALIAS = PARAMS["serving"]["model_alias"]


@pytest.fixture(autouse=True)
def _isolate_mlflow_state(monkeypatch):
    """register() sets the global tracking URI; reset it so it cannot leak into other tests."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield
    mlflow.set_tracking_uri(None)


def _frame(n: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({c: ["A"] * n for c in CATEGORICAL_FEATURES})
    df["soc_code"] = ["15-1132", "15-2041"] * (n // 2)
    df["worksite_state"] = ["CA", "TX"] * (n // 2)
    for c in BOOLEAN_FEATURES:
        df[c] = [True, None, False, True] * (n // 4)  # nulls are real: clean.py lets <NA> through
    df = df.astype(DTYPES)
    df.loc[df.index[1], ["naics_2", "worksite_city"]] = pd.NA  # clean.py can emit NA for both
    df["annual_wage"] = np.linspace(60_000, 150_000, n)
    df["log_wage"] = np.log(df["annual_wage"])
    return df


def test_register_sets_semver_tag_and_alias_with_null_full_time(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    version = register(frame, frame, tracking_uri=uri, artifact_root=tmp_path / "mlruns")

    client = MlflowClient(tracking_uri=uri)
    assert client.get_model_version(MODEL_NAME, version).tags["semver"] == INITIAL_SEMVER
    assert str(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version) == version

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    preds = model.predict(to_model_input(frame))
    assert len(preds) == len(frame)
    assert (np.asarray(preds) > 0).all()


def test_semver_patch_is_bumped_on_each_registration(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    v1 = register(frame, frame, tracking_uri=uri, artifact_root=tmp_path / "mlruns")
    v2 = register(frame, frame, tracking_uri=uri, artifact_root=tmp_path / "mlruns")

    client = MlflowClient(tracking_uri=uri)
    assert client.get_model_version(MODEL_NAME, v1).tags["semver"] == "0.1.0"
    assert client.get_model_version(MODEL_NAME, v2).tags["semver"] == "0.1.1"
    assert str(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version) == v2


def test_alias_not_moved_when_self_check_fails(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    real = register_baseline.regression_report
    calls = {"n": 0}

    def flaky(y_true, y_pred):
        calls["n"] += 1
        out = real(y_true, y_pred)
        return out if calls["n"] == 1 else {k: v + 1 for k, v in out.items()}

    monkeypatch.setattr(register_baseline, "regression_report", flaky)
    frame = _frame()
    with pytest.raises(RuntimeError):
        register(frame, frame, tracking_uri=uri, artifact_root=tmp_path / "mlruns")

    client = MlflowClient(tracking_uri=uri)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(MODEL_NAME, ALIAS)


def test_tracking_server_chooses_artifact_location(tmp_path, monkeypatch):
    """Against a tracking server the experiment must not point at a file:// path on the client."""
    created = {}

    class FakeClient:
        def __init__(self, tracking_uri=None):
            pass

        def get_experiment_by_name(self, name):
            return None

        def create_experiment(self, name, artifact_location=None):
            created["location"] = artifact_location
            raise RuntimeError("stop after create_experiment")

    monkeypatch.setattr(register_baseline, "MlflowClient", FakeClient)
    monkeypatch.setattr(register_baseline.mlflow, "set_tracking_uri", lambda uri: None)
    with pytest.raises(RuntimeError, match="stop after"):
        register(_frame(), _frame(), tracking_uri="http://mlflow:5000", artifact_root=tmp_path)
    assert created["location"] is None
