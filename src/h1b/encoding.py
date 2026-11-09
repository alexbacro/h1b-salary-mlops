"""Turns the h1b.features model input into the frame the AutoML models train on.

Fitted on train only and shipped inside the registered model, so the API keeps sending the raw
feature columns. High-cardinality text columns (job_title, employer_name) become log(1 + train
frequency); a value never seen in train gets 0. Other text columns stay categorical, and values
unseen in train become missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from h1b.features import HIGH_CARDINALITY, LOW_CARDINALITY, MEDIUM_CARDINALITY, to_model_input

CATEGORY_COLUMNS = LOW_CARDINALITY + MEDIUM_CARDINALITY


class FeatureEncoder:
    def fit(self, df: pd.DataFrame) -> FeatureEncoder:
        x = to_model_input(df)
        self.categories_ = {c: pd.Index(x[c].dropna().astype(object).unique()) for c in CATEGORY_COLUMNS}
        self.log_freq_ = {c: np.log1p(x[c].astype(object).value_counts()) for c in HIGH_CARDINALITY}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = to_model_input(df)
        out = pd.DataFrame(index=x.index)
        for c in CATEGORY_COLUMNS:
            out[c] = pd.Categorical(x[c].astype(object), categories=self.categories_[c])
        for c in HIGH_CARDINALITY:
            out[f"{c}_log_freq"] = x[c].astype(object).map(self.log_freq_[c]).fillna(0.0).astype("float64")
        out["full_time"] = x["full_time"]
        return out
