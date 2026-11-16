import numpy as np
import pandas as pd

from h1b.drift_report import drift_scores, flagged
from h1b.drift_simulation import (
    SCENARIOS,
    full_time_shift,
    swap_title_employer,
    unseen_names,
    unseen_soc,
    wage_shift,
)


def _df(n=10):
    return pd.DataFrame(
        {
            "soc_code": ["15-1252"] * n,
            "soc_major": ["15"] * n,
            "job_title": ["ENGINEER"] * n,
            "employer_name": ["ACME"] * n,
            "full_time": ["True"] * n,
            "log_wage": np.full(n, 11.5),
        }
    )


def test_unseen_soc_changes_code_and_derived_major():
    out = unseen_soc(_df(), np.random.default_rng(0), frac=1.0)
    assert set(out["soc_code"]) == {"99-9999"}
    assert set(out["soc_major"]) == {"99"}


def test_unseen_names_are_new_and_unique():
    out = unseen_names(_df(), np.random.default_rng(0), frac=1.0)
    assert out["employer_name"].nunique() == 10
    assert out["job_title"].str.startswith("NEW JOB TITLE").all()


def test_zero_fraction_changes_nothing():
    df = _df()
    for fn in (unseen_soc, unseen_names, full_time_shift):
        pd.testing.assert_frame_equal(fn(df, np.random.default_rng(0), frac=0.0), df)


def test_partial_fraction_changes_about_that_share():
    out = unseen_soc(_df(1000), np.random.default_rng(0), frac=0.3)
    changed = (out["soc_code"] == "99-9999").mean()
    assert 0.25 < changed < 0.35


def test_same_seed_gives_same_corruption():
    a = unseen_names(_df(200), np.random.default_rng(42))
    b = unseen_names(_df(200), np.random.default_rng(42))
    pd.testing.assert_frame_equal(a, b)


def test_full_time_shift():
    out = full_time_shift(_df(), np.random.default_rng(0), frac=1.0)
    assert set(out["full_time"]) == {"False"}


def test_swap_title_employer():
    out = swap_title_employer(_df())
    assert set(out["job_title"]) == {"ACME"}
    assert set(out["employer_name"]) == {"ENGINEER"}


def test_wage_shift_adds_delta_and_keeps_input():
    df = _df()
    out = wage_shift(df, delta=0.3)
    assert np.allclose(out["log_wage"], 11.8)
    assert np.allclose(df["log_wage"], 11.5)


def test_expected_columns_are_among_the_columns_checked():
    for name, (_, columns, expected) in SCENARIOS.items():
        assert expected <= set(columns), name


def test_drift_scores_and_flagged_parse_evidently_names():
    metrics = [
        {"metric_name": "DriftedColumnsCount(columns=a,b,drift_share=0.5)", "value": {"count": 1.0}},
        {"metric_name": "ValueDrift(column=a,method=Jensen-Shannon distance,threshold=0.1)", "value": 0.32},
        {"metric_name": "ValueDrift(column=b,method=Wasserstein distance (normed),threshold=0.1)", "value": 0.02},
    ]
    scores = drift_scores(metrics)
    assert scores == {"a": (0.32, 0.1), "b": (0.02, 0.1)}
    assert flagged(scores) == ["a"]
