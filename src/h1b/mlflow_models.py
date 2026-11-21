"""MLflow pyfunc wrappers so models can be served via `models:/<name>@champion`.

Loading a logged model needs the `h1b` package importable (the serving image
must `pip install -e .`).
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd

from h1b.baseline import MedianBaseline


class BaselinePyfunc(mlflow.pyfunc.PythonModel):
    """Median baseline: takes the feature frame (see h1b.features), returns annual wage in USD."""

    def __init__(self, model: MedianBaseline):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.model.predict(model_input)


class FlamlPyfunc(mlflow.pyfunc.PythonModel):
    """AutoML model trained on log_wage. Returns annual wage in USD, like every registered model (#8, item 2)."""

    def __init__(self, encoder, automl, log_shift: float = 0.0):
        self.encoder = encoder
        self.automl = automl
        self.log_shift = log_shift  # yearly wage growth added back at prediction time (0 = no adjustment)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        # getattr: models registered before log_shift existed (e.g. v2) have no such attribute
        shift = getattr(self, "log_shift", 0.0)
        return np.exp(self.automl.predict(self.encoder.transform(model_input)) + shift)
