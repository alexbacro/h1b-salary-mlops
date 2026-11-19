"""API-level production check: send FY25 holdout batches through the deployed API and monitor them.

The monitoring script is the API client (the API itself logs nothing). For the clean FY25 sample and
for each corruption it
  1. sends the raw records to POST /predict in batches and keeps the predictions,
  2. rebuilds the model input locally with prepare_request, so the features are exactly what the API computed,
  3. runs the drift report on those features against a train sample, plus the unseen-share check,
  4. reports MAE/MAPE of the predictions against the real wages, and the drift of the predictions
     (informational) against predictions on clean val.
The wage shift is not an API input, so it is a target-drift test on the actual wages (no API call).
Clean FY25 is only reported. A corruption passes when every column it must flag is flagged and every
unseen-share alert it must raise fires. Anything else that is flagged is natural drift, reported as
also_flagged. Exit code 1 if a corruption is not caught.

Both sides go through to_model_input, so full_time is 1.0 / 0.0 / missing in the reference and the
current data. This is the FY25 holdout, used here as the production check only; never tune on it.

Usage: python -m h1b.api_check --url http://localhost:8000 --n 20000   (make api-check)
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from h1b.config import get_paths, load_params
from h1b.drift_report import (
    BUCKETED,
    CAT,
    FEATURES,
    NUM,
    _label_missing,
    bucket,
    build_report,
    drift_scores,
    flagged,
    top_values,
    unseen_alerts,
    unseen_share,
    vocabulary,
)
from h1b.features import REQUEST_FEATURES, TARGET, prepare_request, to_model_input

MAX_BATCH = 5_000  # the API accepts up to 10,000 records per request
PRED_COL = "log_prediction"
REFERENCE_ROWS = 50_000  # same size as the reference in drift_report


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def post_json(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned {exc.code}: {exc.read().decode()[:500]}") from exc


def to_records(raw):
    """Raw frame -> JSON-ready records: missing values become None, full_time a real bool."""
    records = []
    for row in raw[REQUEST_FEATURES].itertuples(index=False):
        rec = {}
        for col, value in zip(REQUEST_FEATURES, row, strict=True):
            if pd.isna(value):
                rec[col] = None
            elif col == "full_time":
                rec[col] = bool(value)
            else:
                rec[col] = str(value)
        records.append(rec)
    return records


def predict(url, raw):
    """POST the records in batches; returns (predictions, model name, model version)."""
    records = to_records(raw)
    preds, versions, name = [], set(), None
    for i in range(0, len(records), MAX_BATCH):
        body = post_json(f"{url}/predict", {"records": records[i : i + MAX_BATCH]})
        preds.extend(body["predictions"])
        versions.add(body["model_version"])
        name = body["model_name"]
    if len(versions) != 1:
        raise RuntimeError(f"the champion changed during the run: versions {sorted(versions)}")
    return np.asarray(preds, dtype=float), name, versions.pop()


def performance(pred, actual):
    err = np.abs(pred - actual)
    return {"mae": float(err.mean()), "mape": float((err / actual).mean())}


# Corruptions on the raw request frame: well-formed values, wrong content (they pass API validation).
def corrupt_unseen_soc(raw, rng, frac=0.3):
    out = raw.copy()
    mask = rng.random(len(out)) < frac
    out.loc[mask, "soc_code"] = "99-9999"  # soc_major is derived by the API, never sent
    return out


def corrupt_unseen_names(raw, rng, frac=0.4):
    out = raw.copy()
    mask = rng.random(len(out)) < frac
    if mask.any():
        ids = np.flatnonzero(mask)
        out.loc[mask, "employer_name"] = [f"NEW EMPLOYER {i}" for i in ids]
        out.loc[mask, "job_title"] = [f"NEW JOB TITLE {i}" for i in ids]
    return out


def corrupt_full_time_shift(raw, rng, frac=0.6):
    out = raw.copy()
    mask = rng.random(len(out)) < frac
    out.loc[mask, "full_time"] = False
    return out


def corrupt_swap_title_employer(raw, rng=None):
    out = raw.copy()
    out["job_title"] = raw["employer_name"]
    out["employer_name"] = raw["job_title"]
    return out


# expected = columns that must be flagged (None: report only); alerts = unseen-share alerts that must fire.
SCENARIOS = [
    {"name": "clean_fy25", "corrupt": None, "expected": None, "alerts": None},
    {
        "name": "unseen_soc_30pct",
        "corrupt": corrupt_unseen_soc,
        "expected": {"soc_code", "soc_major"},
        "alerts": {"soc_code"},
    },
    {
        "name": "unseen_names_40pct",
        "corrupt": corrupt_unseen_names,
        "expected": {"employer_name", "job_title"},
        "alerts": {"employer_name", "job_title"},
    },
    {
        "name": "full_time_shift_60pct",
        "corrupt": corrupt_full_time_shift,
        "expected": {"full_time"},
        "alerts": set(),
    },
    {
        "name": "swap_title_employer",
        "corrupt": corrupt_swap_title_employer,
        "expected": {"employer_name", "job_title"},
        "alerts": {"employer_name", "job_title"},
    },
    {
        "name": "wage_shift_plus_0.3",
        "corrupt": None,
        "target_shift": 0.3,
        "expected": {"log_wage"},
        "alerts": set(),
    },
]


def evaluate(hit, alerts, expected, expected_alerts):
    """Pass when everything expected fired; anything extra is natural drift (also_flagged)."""
    if expected is None:
        return {"passed": None, "missing_flags": [], "also_flagged": sorted(hit), "missing_alerts": []}
    missing = sorted(expected - set(hit))
    missing_alerts = sorted(expected_alerts - set(alerts))
    return {
        "passed": not missing and not missing_alerts,
        "missing_flags": missing,
        "also_flagged": sorted(set(hit) - expected),
        "missing_alerts": missing_alerts,
    }


def reference_frame(path, n, seed):
    """Train sample in the same format the API path produces (full_time as 1.0 / 0.0 / missing)."""
    df = pd.read_parquet(path, columns=FEATURES + [TARGET])
    df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    model_input = to_model_input(df)
    model_input[TARGET] = df[TARGET].to_numpy()
    return _label_missing(model_input, CAT)


def prediction_drift(ref_log_pred, cur_log_pred):
    definition = DataDefinition(numerical_columns=[PRED_COL])
    ref_ds = Dataset.from_pandas(pd.DataFrame({PRED_COL: ref_log_pred}), data_definition=definition)
    cur_ds = Dataset.from_pandas(pd.DataFrame({PRED_COL: cur_log_pred}), data_definition=definition)
    snapshot = Report([DataDriftPreset(columns=[PRED_COL])]).run(cur_ds, ref_ds)
    value, threshold = drift_scores(snapshot.dict()["metrics"])[PRED_COL]
    return {"score": round(value, 4), "threshold": threshold, "flagged": value >= threshold}


def baseline_val_mae(reports_dir):
    path = reports_dir / "baseline_metrics_val.json"
    return json.loads(path.read_text()).get("mae") if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=20_000, help="FY25 rows to use (all of them: 566593)")
    args = parser.parse_args()

    params = load_params()
    paths = get_paths(params)
    mon, seed, limits = params["monitoring"], params["seed"], params["monitoring"]["unseen_share_max"]

    try:
        health = get_json(f"{args.url}/health")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach {args.url} ({exc.reason}). Is the stack running? Try: make up")
    if not health.get("model_loaded"):
        sys.exit("the API has no model loaded. Try: make register-docker")

    train_path = paths.processed / "train.parquet"
    ref_raw = reference_frame(train_path, REFERENCE_ROWS, seed)
    keep = top_values(ref_raw, BUCKETED, mon["top_n"])
    ref = bucket(ref_raw, keep)
    vocab = vocabulary(train_path, BUCKETED)

    holdout = pd.read_parquet(paths.holdout / "test.parquet")
    sample = holdout.sample(min(args.n, len(holdout)), random_state=seed).reset_index(drop=True)
    actual = sample["annual_wage"].to_numpy(dtype=float)
    actual_log = sample[TARGET].to_numpy(dtype=float)

    val = pd.read_parquet(paths.processed / "val.parquet", columns=REQUEST_FEATURES)
    val = val.sample(min(args.n, len(val)), random_state=seed).reset_index(drop=True)
    ref_pred, model_name, model_version = predict(args.url, val)
    ref_log_pred = np.log(np.clip(ref_pred, 1.0, None))
    print(f"API model: {model_name} version {model_version}; FY25 rows: {len(sample)}")

    records = []
    for s in SCENARIOS:
        shift = s.get("target_shift", 0.0)
        bad_raw = sample if s["corrupt"] is None else s["corrupt"](sample, np.random.default_rng(seed))

        perf, pred_drift = None, None
        if "target_shift" not in s:
            pred, _, version = predict(args.url, bad_raw)
            if version != model_version:
                raise RuntimeError(f"champion moved from version {model_version} to {version}")
            perf = performance(pred, actual)
            pred_drift = prediction_drift(ref_log_pred, np.log(np.clip(pred, 1.0, None)))

        cur = prepare_request(bad_raw)  # exactly what the API computed from the same records
        cur[TARGET] = actual_log + shift
        cur = _label_missing(cur, CAT)
        unseen = unseen_share(vocab, cur, BUCKETED)
        columns = NUM if "target_shift" in s else FEATURES
        snapshot = build_report(ref, bucket(cur, keep), columns)
        snapshot.save_html(str(paths.reports / f"api_check_{s['name']}.html"))
        scores = drift_scores(snapshot.dict()["metrics"])
        hit = flagged(scores)
        alerts = unseen_alerts(unseen, limits)
        verdict = evaluate(hit, alerts, s["expected"], s["alerts"])
        records.append(
            {
                "scenario": s["name"],
                "api_called": perf is not None,
                "flagged": hit,
                "expected": None if s["expected"] is None else sorted(s["expected"]),
                "unseen_alerts": alerts,
                "expected_unseen_alerts": None if s["alerts"] is None else sorted(s["alerts"]),
                "alert": bool(hit or alerts),
                **verdict,
                "performance": perf,
                "prediction_drift": pred_drift,
                "scores": {c: {"score": round(v, 4), "threshold": t} for c, (v, t) in scores.items()},
                "unseen_share": {c: round(v, 4) for c, v in unseen.items()},
            }
        )
        status = {None: "report only", True: "PASS", False: "FAIL"}[verdict["passed"]]
        print(f"{s['name']}: flagged {', '.join(hit) or '-'} | unseen alerts {', '.join(alerts) or '-'} | {status}")

    failed = [r["scenario"] for r in records if r["passed"] is False]
    summary = {
        "model": {"name": model_name, "version": model_version},
        "n_rows": len(sample),
        "seed": seed,
        "baseline_val_mae": baseline_val_mae(paths.reports),
        "unseen_share_max": limits,
        "all_passed": not failed,
        "scenarios": records,
    }
    out_path = paths.reports / "api_check_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    table = pd.DataFrame(
        [
            {
                "scenario": r["scenario"],
                "flagged": ", ".join(r["flagged"]) or "-",
                "unseen_alerts": ", ".join(r["unseen_alerts"]) or "-",
                "mae": None if r["performance"] is None else round(r["performance"]["mae"]),
                "pred_drift": None if r["prediction_drift"] is None else r["prediction_drift"]["score"],
                "passed": r["passed"],
            }
            for r in records
        ]
    )
    print()
    print(table.to_string(index=False))
    print(f"\nbaseline val MAE: {summary['baseline_val_mae']}   summary: {out_path}")
    if failed:
        print(f"FAILED scenarios: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
