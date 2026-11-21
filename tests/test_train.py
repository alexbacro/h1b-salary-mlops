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
from h1b.register_baseline import register
from h1b.train import _year_adjusted_target, champion_log_rmse, train

PARAMS = load_params()
MODEL_NAME = PARAMS["serving"]["model_name"]
ALIAS = PARAMS["serving"]["model_alias"]
FAST = {**PARAMS, "training": {"time_budget_s": 5, "estimator_list": ["lgbm"], "metric": "mse", "max_train_rows": None}}


@pytest.fixture(autouse=True)
def _isolate_mlflow_state(monkeypatch, tmp_path):
    """train() sets the global tracking URI; reset it so it cannot leak into other tests."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
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


def test_if_better_compares_against_the_registered_baseline(tmp_path):
    """The Airflow path: baseline is @champion (no tag, metric only on its run); train() must compare to it."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    register(frame, frame, tracking_uri=uri, artifact_root=tmp_path / "artifacts")

    client = MlflowClient(tracking_uri=uri)
    has_champion, baseline = champion_log_rmse(client, MODEL_NAME, ALIAS)
    assert has_champion and baseline is not None  # the baseline's metric is found via its run

    result = train(frame, frame, FAST, tracking_uri=uri, promote="if-better")
    assert result["champion_log_rmse"] == pytest.approx(baseline)
    assert result["promoted"] == (result["metrics"]["log_rmse"] < baseline)
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    assert (str(version) == result["version"]) == result["promoted"]


def test_year_adjusted_target_lifts_old_years_and_returns_growth():
    df = pd.DataFrame({"fiscal_year": [2022, 2022, 2023, 2023], "log_wage": [11.0, 11.2, 11.1, 11.3]})
    adjusted, growth = _year_adjusted_target(df)
    assert growth == pytest.approx(0.1)
    assert adjusted.tolist() == pytest.approx([11.1, 11.3, 11.1, 11.3])


def test_train_with_wage_growth_adjustment_registers_a_usd_model(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    frame["fiscal_year"] = [2022, 2023] * (len(frame) // 2)
    frame["log_wage"] = frame["log_wage"] + 0.05 * (frame["fiscal_year"] == 2023)
    params = {**FAST, "training": {**FAST["training"], "wage_growth_adjust": True}}
    result = train(frame, frame, params, tracking_uri=uri, promote="never")  # self-check runs inside train()

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{result['version']}")
    preds = np.asarray(model.predict(to_model_input(frame)))
    assert (preds > 0).all()  # USD, not log


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
