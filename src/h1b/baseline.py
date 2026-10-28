"""Baseline: median wage by (SOC code, worksite state), the bar any model has to beat.

Falls back to the SOC median, then the global median, for combinations not seen in training.
Medians are taken on log wage; because log is monotone this is (up to ties) the log of the median wage.

Usage:
    python -m h1b.baseline                # fit on train, evaluate on val
    python -m h1b.baseline --split test   # production validation only
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from h1b.config import get_paths, load_params
from h1b.metrics import regression_report

log = logging.getLogger(__name__)


class MedianBaseline:
    """Hierarchical median lookup: (soc_code, worksite_state) -> soc_code -> global."""

    def fit(self, df: pd.DataFrame) -> MedianBaseline:
        self.by_soc_state_ = df.groupby(["soc_code", "worksite_state"])["log_wage"].median()
        self.by_soc_ = df.groupby("soc_code")["log_wage"].median()
        self.global_ = float(df["log_wage"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted annual wages in dollars."""
        keys = pd.MultiIndex.from_frame(df[["soc_code", "worksite_state"]])
        pred = pd.Series(self.by_soc_state_.reindex(keys).to_numpy(), index=df.index)
        pred = pred.fillna(df["soc_code"].map(self.by_soc_)).fillna(self.global_)
        return np.exp(pred.to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = get_paths(load_params())

    train = pd.read_parquet(paths.processed / "train.parquet")
    eval_path = paths.processed / "val.parquet" if args.split == "val" else paths.holdout / "test.parquet"
    eval_df = pd.read_parquet(eval_path)

    model = MedianBaseline().fit(train)
    metrics = regression_report(eval_df["annual_wage"], model.predict(eval_df))
    (paths.reports / f"baseline_metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    log.info("baseline on %s: %s", args.split, metrics)


if __name__ == "__main__":
    main()
