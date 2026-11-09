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
