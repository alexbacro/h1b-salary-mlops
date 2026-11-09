"""Feature contract: request columns, model-input dtypes, and API normalization matching clean.py."""

import pandas as pd
import pytest

from h1b.clean import clean
from h1b.features import (
    BOOLEAN_FEATURES,
    DERIVED_FEATURES,
    FEATURES,
    REQUEST_FEATURES,
    prepare_request,
    to_model_input,
)
from tests.conftest import raw_row


def test_request_and_derived_features_partition_features():
    assert set(REQUEST_FEATURES) | set(DERIVED_FEATURES) == set(FEATURES)
    assert not set(REQUEST_FEATURES) & set(DERIVED_FEATURES)


def test_to_model_input_casts_booleans_to_float_and_keeps_nulls():
    df = pd.DataFrame({c: ["X", "Y", "Z"] for c in FEATURES})
    df["full_time"] = [True, False, None]
    out = to_model_input(df)
    assert list(out.columns) == FEATURES
    assert (out[BOOLEAN_FEATURES].dtypes == "float64").all()
    assert out["full_time"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(out["full_time"].iloc[2])


def test_to_model_input_rejects_missing_columns():
    with pytest.raises(ValueError, match="soc_code"):
        to_model_input(pd.DataFrame({c: ["X"] for c in FEATURES if c != "soc_code"}))


def test_prepare_request_normalizes_raw_values_and_derives_soc_major():
    raw = pd.DataFrame(
        [
            {
                "naics_2": "541511",
                "worksite_state": " ca",
                "soc_code": "15-1252.00",
                "worksite_city": "san jose",
                "job_title": "Data Scientist, II ",
                "employer_name": "Acme Corp.",
                "full_time": None,
            }
        ]
    )
    row = prepare_request(raw).iloc[0]
    assert row["soc_code"] == "15-1252"
    assert row["soc_major"] == "15"
    assert row["worksite_state"] == "CA"
    assert row["worksite_city"] == "SAN JOSE"
    assert row["job_title"] == "DATA SCIENTIST II"
    assert row["employer_name"] == "ACME CORP"
    assert row["naics_2"] == "54"
    assert pd.isna(row["full_time"])


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # the conftest default row
        {"JOB_TITLE": "  data   scientist ", "SOC_CODE": "152051", "WORKSITE_STATE": "il "},  # messy formatting
        # A retired SOC 2010 code. Passes today; if clean.py starts remapping 2010 -> 2018 codes (PR #9)
        # and prepare_request does not, this fails, which is exactly the train/serve skew to catch.
        {"SOC_CODE": "15-1132.00"},
    ],
)
def test_prepare_request_matches_clean(filters, overrides):
    """The API path and the training path must produce identical model input for the same raw filing."""
    raw = raw_row(**overrides)
    cleaned, _ = clean(pd.DataFrame([raw]), filters)
    request = pd.DataFrame(
        [
            {
                "naics_2": raw["NAICS_CODE"],
                "worksite_state": raw["WORKSITE_STATE"],
                "soc_code": raw["SOC_CODE"],
                "worksite_city": raw["WORKSITE_CITY"],
                "job_title": raw["JOB_TITLE"],
                "employer_name": raw["EMPLOYER_NAME"],
                "full_time": raw["FULL_TIME_POSITION"] == "Y",
            }
        ]
    )
    pd.testing.assert_frame_equal(prepare_request(request), to_model_input(cleaned))
