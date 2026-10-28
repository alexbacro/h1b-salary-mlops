"""Stage 2: temporal train / validation / test split on fiscal year.

train.parquet and val.parquet go to data/processed/. test.parquet goes to
data/holdout/, a separate directory that training code never reads. It is only
used for production validation (sending it through the deployed API).

Usage:
    python -m h1b.split
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from h1b.config import get_paths, load_params

log = logging.getLogger(__name__)


def temporal_split(df: pd.DataFrame, split_cfg: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    """Partition rows by fiscal_year according to split_cfg {train_years, val_years, test_years}."""
    years = {name: set(split_cfg[f"{name}_years"]) for name in ("train", "val", "test")}
    if (years["train"] & years["val"]) or (years["train"] & years["test"]) or (years["val"] & years["test"]):
        raise ValueError(f"Split years overlap: {split_cfg}")
    if max(years["train"]) >= min(years["val"]) or max(years["val"]) >= min(years["test"]):
        raise ValueError(f"Split must be chronological (train < val < test): {split_cfg}")

    splits = {name: df[df["fiscal_year"].isin(ys)].reset_index(drop=True) for name, ys in years.items()}
    empty = [name for name, part in splits.items() if part.empty]
    if empty:
        raise ValueError(f"Empty split(s) {empty}; check that data/raw has files for {split_cfg}")
    return splits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    paths = get_paths(params)

    df = pd.read_parquet(paths.processed / "lca_clean.parquet")
    splits = temporal_split(df, params["split"])

    splits["train"].to_parquet(paths.processed / "train.parquet", index=False)
    splits["val"].to_parquet(paths.processed / "val.parquet", index=False)
    splits["test"].to_parquet(paths.holdout / "test.parquet", index=False)

    counts = {name: len(part) for name, part in splits.items()}
    (paths.reports / "split_report.json").write_text(json.dumps(counts, indent=2))
    log.info("split sizes: %s", counts)


if __name__ == "__main__":
    main()
