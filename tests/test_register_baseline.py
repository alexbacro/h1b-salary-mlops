"""The baseline registers, gets its semver tag and loads through @champion."""

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from h1b.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, DTYPES, validate_features
from h1b.register_baseline import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, SEMVER, register


def _frame(n: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({c: ["A"] * n for c in CATEGORICAL_FEATURES})
    df["soc_code"] = ["15-1132", "15-2041"] * (n // 2)
    df["worksite_state"] = ["CA", "TX"] * (n // 2)
    for c in BOOLEAN_FEATURES:
        df[c] = True
    df = df.astype(DTYPES)
    df["annual_wage"] = np.linspace(60_000, 150_000, n)
    df["log_wage"] = np.log(df["annual_wage"])
    return df


def test_register_sets_semver_tag_and_champion_alias(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    frame = _frame()
    version = register(frame, frame, tracking_uri=uri, artifact_root=tmp_path / "mlruns")

    client = MlflowClient(tracking_uri=uri)
    assert client.get_model_version(REGISTERED_MODEL_NAME, version).tags["semver"] == SEMVER
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS).version) == version

    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    preds = model.predict(validate_features(frame))
    assert len(preds) == len(frame)
    assert (np.asarray(preds) > 0).all()
