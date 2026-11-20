import json

import numpy as np
import pandas as pd

from h1b.api_check import (
    corrupt_full_time_shift,
    corrupt_swap_title_employer,
    corrupt_unseen_names,
    corrupt_unseen_soc,
    evaluate,
    performance,
    to_records,
)
from h1b.features import REQUEST_FEATURES


def _raw():
    return pd.DataFrame(
        {
            "naics_2": pd.array(["54", None, "54", "54"], dtype="string"),
            "worksite_state": pd.array(["IL"] * 4, dtype="string"),
            "soc_code": pd.array(["15-1252"] * 4, dtype="string"),
            "worksite_city": pd.array(["CHICAGO"] * 4, dtype="string"),
            "job_title": pd.array(["ENGINEER"] * 4, dtype="string"),
            "employer_name": pd.array(["ACME"] * 4, dtype="string"),
            "full_time": pd.array([True, False, None, True], dtype="boolean"),
        }
    )


def test_to_records_turns_missing_into_none_and_keeps_real_bools():
    records = to_records(_raw())
    assert set(records[0]) == set(REQUEST_FEATURES)
    assert records[1]["naics_2"] is None
    assert records[0]["full_time"] is True
    assert records[1]["full_time"] is False
    assert records[2]["full_time"] is None
    json.dumps(records)  # must be JSON-serializable


def test_requests_never_include_the_derived_column():
    assert "soc_major" not in REQUEST_FEATURES
    assert "soc_major" not in corrupt_unseen_soc(_raw(), np.random.default_rng(0), frac=1.0).columns


def test_corruptions():
    rng = np.random.default_rng(0)
    assert set(corrupt_unseen_soc(_raw(), rng, frac=1.0)["soc_code"]) == {"99-9999"}
    names = corrupt_unseen_names(_raw(), rng, frac=1.0)
    assert names["employer_name"].nunique() == 4
    assert names["job_title"].str.startswith("NEW JOB TITLE").all()
    assert set(corrupt_full_time_shift(_raw(), rng, frac=1.0)["full_time"]) == {False}
    swapped = corrupt_swap_title_employer(_raw())
    assert set(swapped["job_title"]) == {"ACME"}
    assert set(swapped["employer_name"]) == {"ENGINEER"}


def test_corruptions_do_not_change_the_input():
    raw = _raw()
    corrupt_unseen_soc(raw, np.random.default_rng(0), frac=1.0)
    assert set(raw["soc_code"]) == {"15-1252"}


def test_evaluate_passes_when_everything_expected_fires_and_reports_extras():
    ok = evaluate(["soc_code", "soc_major", "job_title"], ["soc_code"], {"soc_code", "soc_major"}, {"soc_code"})
    assert ok == {"passed": True, "missing_flags": [], "also_flagged": ["job_title"], "missing_alerts": []}


def test_evaluate_fails_on_missing_flag_or_alert():
    miss_flag = evaluate(["soc_code"], ["soc_code"], {"soc_code", "soc_major"}, {"soc_code"})
    assert miss_flag["passed"] is False and miss_flag["missing_flags"] == ["soc_major"]
    miss_alert = evaluate(["full_time"], [], {"full_time"}, {"job_title"})
    assert miss_alert["passed"] is False and miss_alert["missing_alerts"] == ["job_title"]


def test_evaluate_clean_is_report_only():
    assert evaluate(["job_title"], [], None, None)["passed"] is None


def test_performance():
    out = performance(np.array([110.0, 90.0]), np.array([100.0, 100.0]))
    assert out == {"mae": 10.0, "mape": 0.1}
