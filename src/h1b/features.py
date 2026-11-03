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

# Computed from other columns (see add_derived_features), so API callers never send them.
# Accepting them would let a caller send inconsistent pairs, e.g. soc_major="99" with soc_code="15-1252".
DERIVED_FEATURES: list[str] = ["soc_major"]
# What an API caller sends: every feature except the derived ones.
REQUEST_FEATURES: list[str] = [c for c in FEATURES if c not in DERIVED_FEATURES]

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


def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """validate_features, then boolean columns as float64 (1.0 / 0.0 / NaN).

    MLflow's `boolean` ColSpec cannot hold nulls, so the model signature declares these
    columns as `double` (#8, item 1). Registration, the API and training all call this.
    """
    out = validate_features(df)
    out[BOOLEAN_FEATURES] = out[BOOLEAN_FEATURES].astype("float64")
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add DERIVED_FEATURES from already-normalized columns (same rule as clean.py: SOC major group)."""
    out = df.copy()
    out["soc_major"] = out["soc_code"].str[:2]
    return out


def prepare_request(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw API records (REQUEST_FEATURES) into model input, normalized exactly as clean.py does.

    Without this, '15-1252.00', 'ca' or 'Data Scientist' would not match the training keys and
    would silently fall back to the global median (#8, item 4).
    """
    # Imported here, not at the top: clean.py can then import add_derived_features from this
    # module later without creating a circular import.
    from h1b.clean import normalize_soc, normalize_text

    out = df.copy()
    out["soc_code"] = normalize_soc(out["soc_code"])
    out["worksite_state"] = out["worksite_state"].astype("string").str.upper().str.strip()
    for col in ["job_title", "employer_name", "worksite_city"]:
        out[col] = normalize_text(out[col])
    # Same rule as clean.py: keep the 2-digit NAICS sector, so '541511' and '54' both give '54'.
    out["naics_2"] = out["naics_2"].astype("string").str.extract(r"^(\d{2})")[0]
    return to_model_input(add_derived_features(out))
