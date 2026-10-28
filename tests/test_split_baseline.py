import numpy as np
import pandas as pd
import pytest

from h1b.baseline import MedianBaseline
from h1b.metrics import log_rmse, mae, mape
from h1b.split import temporal_split

SPLIT = {"train_years": [2022, 2023], "val_years": [2024], "test_years": [2025]}


def frame(years) -> pd.DataFrame:
    return pd.DataFrame({"fiscal_year": years, "x": range(len(years))})


def test_temporal_split_partitions_by_year():
    splits = temporal_split(frame([2021, 2022, 2023, 2024, 2025, 2025]), SPLIT)
    assert {k: len(v) for k, v in splits.items()} == {"train": 2, "val": 1, "test": 2}


@pytest.mark.parametrize(
    "cfg",
    [
        {"train_years": [2022, 2024], "val_years": [2024], "test_years": [2025]},  # overlap
        {"train_years": [2025], "val_years": [2024], "test_years": [2023]},  # not chronological
    ],
)
def test_temporal_split_rejects_bad_config(cfg):
    with pytest.raises(ValueError):
        temporal_split(frame([2022, 2023, 2024, 2025]), cfg)


def test_temporal_split_rejects_empty_split():
    with pytest.raises(ValueError, match="test"):
        temporal_split(frame([2022, 2024]), SPLIT)


def test_median_baseline_falls_back_hierarchically():
    train = pd.DataFrame(
        {
            "soc_code": ["15-1252", "15-1252", "15-1252", "13-2011"],
            "worksite_state": ["CA", "CA", "TX", "NY"],
            "annual_wage": [150000, 170000, 100000, 80000],
        }
    )
    train["log_wage"] = np.log(train["annual_wage"])
    model = MedianBaseline().fit(train)

    test = pd.DataFrame({"soc_code": ["15-1252", "15-1252", "99-9999"], "worksite_state": ["TX", "WA", "CA"]})
    pred = model.predict(test)
    assert pred[0] == pytest.approx(100000)  # exact (soc, state) match
    assert pred[1] == pytest.approx(150000)  # soc-only fallback: median of 100k, 150k, 170k
    assert pred[2] == pytest.approx(np.exp(np.median(train["log_wage"])))  # global fallback


def test_metrics():
    y, p = [100.0, 200.0], [110.0, 180.0]
    assert mae(y, p) == pytest.approx(15)
    assert mape(y, p) == pytest.approx(0.1)
    assert log_rmse(y, y) == 0
