import numpy as np
import pandas as pd

from h1b.drift_report import CAT, COLS, load


def _write_parquet(tmp_path, n=100):
    df = pd.DataFrame(
        {
            "soc_major": ["15"] * n,
            "worksite_state": ["IL"] * n,
            "full_time": [True] * n,
            "naics_2": ["54"] * (n - 1) + [np.nan],
            "log_wage": np.linspace(10.5, 12.0, n),
            "case_number": [str(i) for i in range(n)],  # not used for drift
        }
    )
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)
    return path


def test_load_keeps_only_drift_columns(tmp_path):
    df = load(_write_parquet(tmp_path))
    assert set(df.columns) == set(COLS)


def test_load_caps_sample_size(tmp_path):
    path = _write_parquet(tmp_path, n=100)
    assert len(load(path, n=10)) == 10
    assert len(load(path, n=1_000)) == 100


def test_load_makes_categoricals_strings_without_missing(tmp_path):
    df = load(_write_parquet(tmp_path))
    for col in CAT:
        assert df[col].map(type).eq(str).all()
    assert not df[CAT].isna().any().any()
