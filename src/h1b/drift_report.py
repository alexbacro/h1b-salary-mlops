"""First Evidently data drift report: train (FY22-23) vs val (FY24)."""

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from h1b.config import get_paths, load_params

CAT = ["soc_major", "worksite_state", "full_time", "naics_2"]
NUM = ["log_wage"]
COLS = CAT + NUM


def load(path, n=50_000):
    df = pd.read_parquet(path, columns=COLS)
    df = df.sample(min(n, len(df)), random_state=42)
    df[CAT] = df[CAT].astype(str)
    return df


def main():
    paths = get_paths(load_params())
    ref = load(paths.processed / "train.parquet")  # FY22-23
    cur = load(paths.processed / "val.parquet")  # FY24
    definition = DataDefinition(numerical_columns=NUM, categorical_columns=CAT)
    ref_ds = Dataset.from_pandas(ref, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur, data_definition=definition)
    report = Report([DataDriftPreset()])
    snapshot = report.run(cur_ds, ref_ds)  # current first, reference second
    out = paths.reports / "drift_train_vs_val.html"
    snapshot.save_html(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
