"""FLAML training registers a USD-returning model, promotes only when allowed, and encodes unseen values safely."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from h1b.config import load_params
from h1b.encoding import FeatureEncoder
from h1b.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, DTYPES, to_model_input
from h1b.train import train

PARAMS = load_params()
MODEL_NAME = PARAMS["serving"]["model_name"]
ALIAS = PARAMS["serving"]["model_alias"]
FAST = {**PARAMS, "training": {"time_budget_s": 5, "estimator_list": ["lgbm"], "metric": "mse", "max_train_rows": None}}


@pytest.fixture(autouse=True)
def _isolate_mlflow_state(monkeypatch):
    """train() sets the global tracking URI; reset it so it cannot leak into other tests."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield
    mlflow.set_tracking_uri(None)


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    soc = rng.choice(["15-1252", "15-1211", "13-2011"], n)
    state = rng.choice(["CA", "TX", "NY"], n)
    df = pd.DataFrame({c: ["A"] * n for c in CATEGORICAL_FEATURES})
    df["soc_code"] = soc
    df["soc_major"] = [s[:2] for s in soc]
    df["worksite_state"] = state
    df["job_title"] = rng.choice(["ENGINEER", "ANALYST", "MANAGER", "SCIENTIST"], n)
    df["employer_name"] = rng.choice(["ACME", "GLOBEX", "INITECH"], n)
    for c in BOOLEAN_FEATURES:
        df[c] = [[True, None, False][i % 3] for i in range(n)]
    df = df.astype(DTYPES)
    df["log_wage"] = 11 + 0.3 * (soc == "15-1252") + 0.1 * (state == "CA") + rng.normal(0, 0.05, n)
    df["annual_wage"] = np.exp(df["log_wage"])
    return df


def test_train_registers_and_promotes_when_there_is_no_champion(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    result = train(frame, frame, FAST, tracking_uri=uri)
    assert result["promoted"] is True

    client = MlflowClient(tracking_uri=uri)
    assert str(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version) == result["version"]
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    preds = np.asarray(model.predict(to_model_input(frame)))
    assert len(preds) == len(frame) and (preds > 0).all()  # USD, not log


def test_promote_never_registers_without_moving_the_alias(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    result = train(frame, frame, FAST, tracking_uri=uri, promote="never")
    assert result["promoted"] is False

    client = MlflowClient(tracking_uri=uri)
    assert client.get_model_version(MODEL_NAME, result["version"]).tags["semver"]
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(MODEL_NAME, ALIAS)


def test_encoder_handles_values_unseen_in_train():
    frame = _frame()
    encoder = FeatureEncoder().fit(frame)
    new = frame.head(2).copy()
    new["employer_name"] = pd.array(["NEVER SEEN", "ACME"], dtype="string")
    new["soc_code"] = pd.array(["99-9999", "15-1252"], dtype="string")
    out = encoder.transform(new)
    assert out["employer_name_log_freq"].iloc[0] == 0.0
    assert out["employer_name_log_freq"].iloc[1] > 0.0
    assert out["soc_code"].isna().tolist() == [True, False]
