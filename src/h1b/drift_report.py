"""Evidently drift reports: train (FY22-23) vs val (FY24).

Columns come from h1b.features so monitoring matches training and the API.
Low-cardinality columns are tested directly. Medium/high-cardinality columns are bucketed
(top N train values, everything else -> "other"), and the share of values never seen in
training is checked against the limits in params.yaml (monitoring.unseen_share_max).

Decision rule (quote this in the README and slides): a column is flagged when its drift score
is at or above its Evidently threshold (0.1 for Jensen-Shannon and normed Wasserstein). That is
stricter than Evidently's dataset-level drift_share=0.5.
"""

import json
import re

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
TOP_N = 50  # default; main() reads monitoring.top_n from params.yaml
VALUE_DRIFT = re.compile(r"ValueDrift\(column=([^,]+),.*threshold=([\d.]+)\)")


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


def unseen_alerts(unseen, limits):
    """Columns whose unseen share is over their limit (columns without a limit never alert)."""
    return sorted(c for c, limit in limits.items() if unseen.get(c, 0.0) > limit)


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


def drift_scores(metrics):
    """{column: (score, threshold)} from Evidently's ValueDrift metrics."""
    out = {}
    for m in metrics:
        match = VALUE_DRIFT.search(m["metric_name"])
        if match:
            out[match[1]] = (float(m["value"]), float(match[2]))
    if not out:
        raise RuntimeError("no ValueDrift metrics parsed; did Evidently change its metric names?")
    return out


def flagged(scores):
    """Columns whose score is at or above their threshold (the decision rule above)."""
    return sorted(c for c, (value, threshold) in scores.items() if value >= threshold)


def main():
    params = load_params()
    paths = get_paths(params)
    mon = params["monitoring"]
    limits = mon["unseen_share_max"]
    train_path = paths.processed / "train.parquet"  # FY22-23
    ref = load(train_path)
    cur = load(paths.processed / "val.parquet")  # FY24

    unseen = unseen_share(vocabulary(train_path, BUCKETED), cur, BUCKETED)
    alerts = unseen_alerts(unseen, limits)
    print("share of val rows with a value never seen in train:")
    for col, share in unseen.items():
        limit = f" (limit {limits[col]})" if col in limits else ""
        print(f"   {col}: {share:.3f}{limit}")
    print("   unseen-share alerts:", ", ".join(alerts) or "none")

    keep = top_values(ref, BUCKETED, mon["top_n"])
    ref, cur = bucket(ref, keep), bucket(cur, keep)

    summary = {
        "reference": "train (FY22-23)",
        "current": "val (FY24)",
        "top_n": mon["top_n"],
        "unseen_share": {c: round(v, 4) for c, v in unseen.items()},
        "unseen_share_max": limits,
        "unseen_alerts": alerts,
    }
    for name, columns in [("features", FEATURES), ("target", NUM)]:
        snapshot = build_report(ref, cur, columns)
        out = paths.reports / f"drift_{name}_train_vs_val.html"
        snapshot.save_html(str(out))
        scores = drift_scores(snapshot.dict()["metrics"])
        hit = flagged(scores)
        summary[name] = {
            "scores": {c: {"score": round(v, 4), "threshold": t} for c, (v, t) in scores.items()},
            "flagged": hit,
        }
        print(f"saved {out}")
        for col, (value, threshold) in scores.items():
            print(f"   {col}: {value:.3f} (threshold {threshold})")
        print("   flagged:", ", ".join(hit) or "none")

    json_path = paths.reports / "drift_train_vs_val.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
