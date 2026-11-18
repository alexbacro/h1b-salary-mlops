"""Drift simulation: corrupt a copy of val and check that Evidently catches it.

Reference = train sample, current = val sample, as in drift_report. Each scenario corrupts the
current data in a way that would pass input validation (well-formed values, wrong content).
Reports go to reports/drift_sim_<scenario>.html and a summary to reports/drift_simulation_summary.csv.
"""

import re

import numpy as np
import pandas as pd

from h1b.config import get_paths, load_params
from h1b.drift_report import (
    BUCKETED,
    FEATURES,
    NUM,
    bucket,
    build_report,
    load,
    top_values,
    unseen_share,
    vocabulary,
)

SEED = 42
VALUE_DRIFT = re.compile(r"ValueDrift\(column=([^,]+),.*threshold=([\d.]+)\)")


def no_corruption(df, rng=None):
    return df.copy()


def unseen_soc(df, rng, frac=0.3):
    """Well-formed SOC codes that never appear in training; soc_major is derived from them."""
    out = df.copy()
    mask = rng.random(len(out)) < frac
    out.loc[mask, "soc_code"] = "99-9999"
    out.loc[mask, "soc_major"] = "99"
    return out


def unseen_names(df, rng, frac=0.4):
    """Employers and job titles that never appear in training."""
    out = df.copy()
    mask = rng.random(len(out)) < frac
    if mask.any():
        ids = np.flatnonzero(mask)
        out.loc[mask, "employer_name"] = [f"NEW EMPLOYER {i}" for i in ids]
        out.loc[mask, "job_title"] = [f"NEW JOB TITLE {i}" for i in ids]
    return out


def full_time_shift(df, rng, frac=0.6):
    """Most training rows are full time; flip a large share to part time."""
    out = df.copy()
    mask = rng.random(len(out)) < frac
    out.loc[mask, "full_time"] = "False"
    return out


def swap_title_employer(df, rng=None):
    """Swap two free-text columns: still well-formed strings, wrong content."""
    out = df.copy()
    out["job_title"] = df["employer_name"]
    out["employer_name"] = df["job_title"]
    return out


def wage_shift(df, rng=None, delta=0.3):
    """Shift the target: log_wage + 0.3 is about 35% higher wages."""
    out = df.copy()
    out["log_wage"] = out["log_wage"] + delta
    return out


# name -> (corruption, columns the report checks)
SCENARIOS = {
    "clean": (no_corruption, FEATURES),
    "unseen_soc_30pct": (unseen_soc, FEATURES),
    "unseen_names_40pct": (unseen_names, FEATURES),
    "full_time_shift_60pct": (full_time_shift, FEATURES),
    "swap_title_employer": (swap_title_employer, FEATURES),
    "wage_shift_plus_0.3": (wage_shift, NUM),
}


def drift_scores(metrics):
    """{column: (score, threshold)} from Evidently's ValueDrift metrics."""
    out = {}
    for m in metrics:
        match = VALUE_DRIFT.search(m["metric_name"])
        if match:
            out[match[1]] = (float(m["value"]), float(match[2]))
    return out


def flagged(scores):
    """Columns whose score is at or above their threshold."""
    return sorted(c for c, (value, threshold) in scores.items() if value >= threshold)


def main():
    paths = get_paths(load_params())
    train_path = paths.processed / "train.parquet"  # FY22-23
    ref_raw = load(train_path)
    cur_raw = load(paths.processed / "val.parquet")  # FY24
    vocab = vocabulary(train_path, BUCKETED)
    keep = top_values(ref_raw, BUCKETED)
    ref = bucket(ref_raw, keep)

    rows = []
    for name, (corrupt, columns) in SCENARIOS.items():
        bad = corrupt(cur_raw, np.random.default_rng(SEED))
        unseen = unseen_share(vocab, bad, BUCKETED)
        snapshot = build_report(ref, bucket(bad, keep), columns)
        snapshot.save_html(str(paths.reports / f"drift_sim_{name}.html"))
        scores = drift_scores(snapshot.dict()["metrics"])
        hit = flagged(scores)
        rows.append(
            {
                "scenario": name,
                "flagged": ", ".join(hit) or "-",
                "n_flagged": len(hit),
                "max_score": round(max(v for v, _ in scores.values()), 3),
                "unseen_soc_code": round(unseen["soc_code"], 3),
                "unseen_job_title": round(unseen["job_title"], 3),
                "unseen_employer": round(unseen["employer_name"], 3),
            }
        )
        print(f"{name}: flagged {rows[-1]['flagged']} (max score {rows[-1]['max_score']})")

    summary = pd.DataFrame(rows)
    summary.to_csv(paths.reports / "drift_simulation_summary.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
