import numpy as np
import pandas as pd

from h1b.drift_report import (
    CAT,
    COLS,
    bucket,
    load,
    top_values,
    unseen_share,
    vocabulary,
)


def _write_parquet(tmp_path, n=100):
    data = {c: ["a"] * n for c in CAT}
    data["full_time"] = [True] * n
    data["naics_2"] = ["54"] * (n - 1) + [np.nan]
    data["log_wage"] = np.linspace(10.5, 12.0, n)
    data["case_number"] = [str(i) for i in range(n)]  # not used for drift
    path = tmp_path / "sample.parquet"
    pd.DataFrame(data).to_parquet(path)
    return path


def test_load_keeps_only_drift_columns(tmp_path):
    df = load(_write_parquet(tmp_path))
    assert set(df.columns) == set(COLS)


def test_load_caps_sample_size(tmp_path):
    path = _write_parquet(tmp_path, n=100)
    assert len(load(path, n=10)) == 10
    assert len(load(path, n=1_000)) == 100


def test_load_labels_missing_categories(tmp_path):
    df = load(_write_parquet(tmp_path))
    assert "missing" in set(df["naics_2"])
    assert "nan" not in set(df["naics_2"])
    for col in CAT:
        assert df[col].map(type).eq(str).all()


def test_bucket_groups_rare_values_into_other():
    ref = pd.DataFrame({"employer_name": ["A"] * 5 + ["B", "C"]})
    keep = top_values(ref, ["employer_name"], n=1)
    out = bucket(pd.DataFrame({"employer_name": ["A", "B", "Z"]}), keep)
    assert out["employer_name"].tolist() == ["A", "other", "other"]


def test_unseen_share_counts_values_missing_from_vocabulary():
    vocab = {"employer_name": {"A", "B"}}
    cur = pd.DataFrame({"employer_name": ["A", "B", "Z", "Y"]})
    assert unseen_share(vocab, cur, ["employer_name"]) == {"employer_name": 0.5}


def test_vocabulary_includes_missing_label(tmp_path):
    vocab = vocabulary(_write_parquet(tmp_path), ["naics_2"])
    assert vocab["naics_2"] == {"54", "missing"}
