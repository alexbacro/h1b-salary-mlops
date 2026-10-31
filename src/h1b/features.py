"""Single source of truth for the model's feature contract.

Training, the FastAPI service, and the drift report all import from here
so the column lists cannot drift apart.

Column names/dtypes match data/processed/{train,val}.parquet
(produced by `make data`).
"""

from __future__ import annotations

import pandas as pd

TARGET = "log_wage"  # ln(annual_wage); regression target

# Grouped by cardinality (train split) to guide encoding at training time
# and drift monitoring.
LOW_CARDINALITY: list[str] = [
    "soc_major",  # 25
    "naics_2",  # 25
    "worksite_state",  # 55
]
MEDIUM_CARDINALITY: list[str] = [
    "soc_code",  # 580
    "worksite_city",  # ~9.6k
]
HIGH_CARDINALITY: list[str] = [
    "job_title",  # ~174k
    "employer_name",  # ~78k
]
CATEGORICAL_FEATURES: list[str] = LOW_CARDINALITY + MEDIUM_CARDINALITY + HIGH_CARDINALITY
BOOLEAN_FEATURES: list[str] = ["full_time"]
NUMERIC_FEATURES: list[str] = []  # none yet; add engineered features here

FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES

# Kept for splitting/bookkeeping; never fed to the model.
NON_FEATURE_COLUMNS: list[str] = [
    "case_number",
    "decision_date",
    "fiscal_year",
    "annual_wage",
]

# Never ingested: leak the target (employers must pay >= prevailing wage).
LEAKAGE_COLUMNS: list[str] = ["PREVAILING_WAGE", "PW_UNIT_OF_PAY", "PW_WAGE_LEVEL"]

DTYPES: dict[str, str] = {
    **{c: "string" for c in CATEGORICAL_FEATURES},
    **{c: "float64" for c in NUMERIC_FEATURES},
    **{c: "boolean" for c in BOOLEAN_FEATURES},
}


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df[FEATURES] with enforced dtypes; raise if columns are missing."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURES].astype(DTYPES)
