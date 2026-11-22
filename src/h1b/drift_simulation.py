"""Drift simulation: corrupt a copy of val and check that Evidently catches it.

Reference = train sample, current = val sample, as in drift_report. Each scenario corrupts the
current data in a way that would pass input validation (well-formed values, wrong content) and
declares which columns the drift report must flag and which unseen-share alerts must fire.
A scenario passes only if exactly those happen, so "monitoring catches it" is checked, not just
read off a table. Seed, top-N and the unseen-share limits come from params.yaml.

Outputs: reports/drift_sim_<scenario>.html, drift_simulation_summary.csv and .json.
Exit code 1 if any scenario fails.
"""

import json
import sys

import numpy as np
import pandas as pd

from h1b.config import get_paths, load_params
from h1b.drift_report import (
    BUCKETED,
    FEATURES,
    NUM,
    bucket,
    build_report,
    drift_scores,
    flagged,
    load,
    top_values,
    unseen_alerts,
    unseen_share,
    vocabulary,
)


def no_corruption(df, rng=None):
    return df.copy()


def unseen_soc(df, rng, frac=0.3):
    """Well-formed SOC codes that never appear in training; soc_major is derived from them.

    All corrupted rows get one new code, so the report shows a jump in "other" for soc_code
    and a new "99" category for soc_major.
    """
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


# name -> (corruption, columns the report checks, columns it must flag, unseen-share alerts it must raise)
SCENARIOS = {
    "clean": (no_corruption, FEATURES, set(), set()),
    "unseen_soc_30pct": (unseen_soc, FEATURES, {"soc_code", "soc_major"}, {"soc_code"}),
    "unseen_names_40pct": (
        unseen_names,
        FEATURES,
        {"employer_name", "job_title"},
        {"employer_name", "job_title"},
    ),
    "full_time_shift_60pct": (full_time_shift, FEATURES, {"full_time"}, set()),
    "swap_title_employer": (
        swap_title_employer,
        FEATURES,
        {"employer_name", "job_title"},
        {"employer_name", "job_title"},
    ),
    "wage_shift_plus_0.3": (wage_shift, NUM, {"log_wage"}, set()),
}


def main():
    params = load_params()
    paths = get_paths(params)
    mon = params["monitoring"]
    seed, limits = params["seed"], mon["unseen_share_max"]
    train_path = paths.processed / "train.parquet"  # FY22-23
    ref_raw = load(train_path)
    cur_raw = load(paths.processed / "val.parquet")  # FY24
    vocab = vocabulary(train_path, BUCKETED)
    keep = top_values(ref_raw, BUCKETED, mon["top_n"])
    ref = bucket(ref_raw, keep)

    records = []
    for name, (corrupt, columns, expected, expected_alerts) in SCENARIOS.items():
        bad = corrupt(cur_raw, np.random.default_rng(seed))
        unseen = unseen_share(vocab, bad, BUCKETED)
        snapshot = build_report(ref, bucket(bad, keep), columns)
        snapshot.save_html(str(paths.reports / f"drift_sim_{name}.html"))
        scores = drift_scores(snapshot.dict()["metrics"])
        hit = flagged(scores)
        alerts = unseen_alerts(unseen, limits)
        passed = set(hit) == expected and set(alerts) == expected_alerts
        records.append(
            {
                "scenario": name,
                "flagged": hit,
                "expected": sorted(expected),
                "unseen_alerts": alerts,
                "expected_unseen_alerts": sorted(expected_alerts),
                "passed": passed,
                "max_score": round(max(v for v, _ in scores.values()), 3),
                "scores": {c: {"score": round(v, 4), "threshold": t} for c, (v, t) in scores.items()},
                "unseen_share": {c: round(v, 4) for c, v in unseen.items()},
            }
        )
        status = (
            "PASS"
            if passed
            else f"FAIL (expected flags {sorted(expected) or 'none'}, "
            f"expected unseen alerts {sorted(expected_alerts) or 'none'})"
        )
        print(f"{name}: flagged {', '.join(hit) or '-'} (max score {records[-1]['max_score']}) {status}")

    failed = [r["scenario"] for r in records if not r["passed"]]
    summary = {
        "seed": seed,
        "top_n": mon["top_n"],
        "unseen_share_max": limits,
        "all_passed": not failed,
        "scenarios": records,
    }
    (paths.reports / "drift_simulation_summary.json").write_text(json.dumps(summary, indent=2))
    rows = [
        {
            "scenario": r["scenario"],
            "flagged": ", ".join(r["flagged"]) or "-",
            "n_flagged": len(r["flagged"]),
            "max_score": r["max_score"],
            "unseen_soc_code": r["unseen_share"]["soc_code"],
            "unseen_job_title": r["unseen_share"]["job_title"],
            "unseen_employer": r["unseen_share"]["employer_name"],
            "unseen_alerts": ", ".join(r["unseen_alerts"]) or "-",
            "passed": r["passed"],
        }
        for r in records
    ]
    table = pd.DataFrame(rows)
    table.to_csv(paths.reports / "drift_simulation_summary.csv", index=False)
    print()
    print(table.to_string(index=False))

    if failed:
        print(f"\nFAILED scenarios: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
