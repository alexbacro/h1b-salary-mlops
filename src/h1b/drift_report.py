"""Evidently drift reports: train (FY22-23) vs val (FY24).

Columns come from h1b.features so monitoring matches training and the API.
Low-cardinality columns are tested directly. Medium/high-cardinality columns
are bucketed (top N values from the reference, everything else -> "other"),
and the share of values never seen in training is reported separately.
"""

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from h1b.config import get_paths, load_params
from h1b.features import (
    BOOLEAN_FEATURES,
    HIGH_CARDINALITY,
    LOW_CARDINALITY,
    MEDIUM_CARDINALITY,
    TARGET,
)

BUCKETED = MEDIUM_CARDINALITY + HIGH_CARDINALITY
CAT = LOW_CARDINALITY + BUCKETED + BOOLEAN_FEATURES
NUM = [TARGET]
COLS = CAT + NUM
FEATURES = CAT  # input drift: are we seeing different kinds of jobs?
TOP_N = 50


def _label_missing(df, columns):
    """Cast to plain strings, labelling missing values 'missing'."""
    out = df.copy()
    for col in columns:
        s = out[col]
        out[col] = s.astype(object).where(s.notna(), "missing").astype(str)
    return out


def load(path, n=50_000):
    df = pd.read_parquet(path, columns=COLS)
    df = df.sample(min(n, len(df)), random_state=42)
    return _label_missing(df, CAT)


def vocabulary(path, columns):
    """All values seen in the full reference file (not just the sample)."""
    full = _label_missing(pd.read_parquet(path, columns=columns), columns)
    return {c: set(full[c]) for c in columns}


def unseen_share(vocab, cur, columns):
    """Share of current rows whose value never appeared in the reference."""
    return {c: float((~cur[c].isin(vocab[c])).mean()) for c in columns}


def top_values(ref, columns, n=TOP_N):
    return {c: set(ref[c].value_counts().head(n).index) for c in columns}


def bucket(df, keep):
    df = df.copy()
    for col, values in keep.items():
        df[col] = df[col].where(df[col].isin(values), "other")
    return df


def build_report(ref, cur, columns):
    definition = DataDefinition(numerical_columns=NUM, categorical_columns=CAT)
    ref_ds = Dataset.from_pandas(ref, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur, data_definition=definition)
    report = Report([DataDriftPreset(columns=columns)])
    return report.run(cur_ds, ref_ds)  # current first, reference second


def main():
    paths = get_paths(load_params())
    train_path = paths.processed / "train.parquet"  # FY22-23
    ref = load(train_path)
    cur = load(paths.processed / "val.parquet")  # FY24

    unseen = unseen_share(vocabulary(train_path, BUCKETED), cur, BUCKETED)
    print("share of val rows with a value never seen in train:")
    for col, share in unseen.items():
        print(f"   {col}: {share:.3f}")

    keep = top_values(ref, BUCKETED)
    ref, cur = bucket(ref, keep), bucket(cur, keep)

    for name, columns in [("features", FEATURES), ("target", NUM)]:
        snapshot = build_report(ref, cur, columns)
        out = paths.reports / f"drift_{name}_train_vs_val.html"
        snapshot.save_html(str(out))
        print(f"saved {out}")
        for metric in snapshot.dict()["metrics"]:
            print("  ", metric["metric_name"], "->", metric["value"])


if __name__ == "__main__":
    main()
