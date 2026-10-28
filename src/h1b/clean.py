"""Stage 1b: combine interim Parquets, normalize wages and text, validate, and write one clean dataset.

Output: data/processed/lca_clean.parquet with one row per certified H-1B case and
the target columns `annual_wage` and `log_wage`. A JSON report of how many rows
each rule removed is written to reports/ingest_report.json.

Usage:
    python -m h1b.clean
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from h1b.config import get_paths, load_params

log = logging.getLogger(__name__)

# Multipliers to convert a wage to a yearly amount (2080 = 40 h x 52 weeks, the DOL convention).
ANNUAL_MULTIPLIER = {
    "HOUR": 2080,
    "WEEK": 52,
    "BIWEEKLY": 26,
    "MONTH": 12,
    "YEAR": 1,
}

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "GU", "VI", "MP", "AS",
}  # fmt: skip

# Columns written to the clean dataset (features + target + metadata), in order.
OUTPUT_COLUMNS = [
    "case_number",
    "decision_date",
    "fiscal_year",
    "job_title",
    "soc_code",
    "soc_major",
    "employer_name",
    "worksite_city",
    "worksite_state",
    "full_time",
    "naics_2",
    "annual_wage",
    "log_wage",
]


# ── Field normalizers ─────────────────────────────────────────────────────────


def normalize_text(s: pd.Series) -> pd.Series:
    """Upper-case, drop punctuation, collapse whitespace. Empty strings become NA."""
    out = (
        s.astype("string")
        .str.upper()
        .str.replace(r"[^A-Z0-9&/ ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out.mask(out == "")


def normalize_soc(s: pd.Series) -> pd.Series:
    """Coerce SOC codes to 'NN-NNNN' ('15-1252.00' and '151252' both -> '15-1252'); invalid -> NA."""
    digits = s.astype("string").str.replace(r"\.\d+$", "", regex=True).str.replace(r"\D", "", regex=True)
    valid = digits.str.len() == 6
    return (digits.str[:2] + "-" + digits.str[2:]).where(valid)


def parse_wage(s: pd.Series) -> pd.Series:
    """'$85,000.00' -> 85000.0. For ranges like '85000 - 95000' the first number is used."""
    first_number = s.astype("string").str.replace(",", "", regex=False).str.extract(r"(\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(first_number, errors="coerce")


def normalize_unit(s: pd.Series) -> pd.Series:
    """'Bi-Weekly' -> 'BIWEEKLY', 'Year' -> 'YEAR'."""
    return s.astype("string").str.upper().str.replace(r"[^A-Z]", "", regex=True)


def fiscal_year(dates: pd.Series) -> pd.Series:
    """US federal fiscal year: Oct 1 2024 - Sep 30 2025 is FY2025."""
    return (dates.dt.year + (dates.dt.month >= 10).astype(int)).astype("Int64")


def normalize_status(s: pd.Series) -> pd.Series:
    """'Certified - Withdrawn' -> 'CERTIFIED-WITHDRAWN'."""
    return s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)


# ── Main cleaning routine ─────────────────────────────────────────────────────


def clean(raw: pd.DataFrame, filters: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all filters/normalizations. Returns (clean_df, report) where report counts rows dropped per rule."""
    report: dict[str, Any] = {"rows_in": len(raw), "dropped": {}}
    df = raw.copy()

    def drop(mask: pd.Series, rule: str) -> None:
        """Remove rows where mask is True and record how many were removed."""
        nonlocal df
        n = int(mask.sum())
        report["dropped"][rule] = n
        df = df.loc[~mask]

    # 1. Dedupe: quarterly files can repeat a case; keep its latest decision.
    df["decision_date"] = pd.to_datetime(df["DECISION_DATE"], errors="coerce")
    drop(df["decision_date"].isna(), "missing_decision_date")
    df = df.sort_values("decision_date")
    drop(df.duplicated("CASE_NUMBER", keep="last"), "duplicate_case_number")

    # 2. Scope: H-1B only, certified only.
    visa = df["VISA_CLASS"].astype("string").str.upper().str.strip()
    drop(~visa.isin([v.upper() for v in filters["visa_classes"]]), "visa_class_out_of_scope")
    drop(~normalize_status(df["CASE_STATUS"]).isin(filters["case_statuses"]), "case_status_not_certified")

    # 3. Wage -> annual. Unknown units are dropped rather than guessed.
    df["multiplier"] = normalize_unit(df["WAGE_UNIT_OF_PAY"]).map(ANNUAL_MULTIPLIER)
    df["wage"] = parse_wage(df["WAGE_RATE_OF_PAY_FROM"])
    drop(df["multiplier"].isna(), "unknown_wage_unit")
    drop(df["wage"].isna(), "unparseable_wage")
    df["annual_wage"] = (df["wage"] * df["multiplier"]).astype(float)
    out_of_range = ~df["annual_wage"].between(filters["min_annual_wage"], filters["max_annual_wage"])
    drop(out_of_range, "annual_wage_out_of_range")
    df["log_wage"] = np.log(df["annual_wage"])

    # 4. Features.
    df["soc_code"] = normalize_soc(df["SOC_CODE"])
    drop(df["soc_code"].isna(), "invalid_soc_code")
    df["soc_major"] = df["soc_code"].str[:2]

    df["worksite_state"] = df["WORKSITE_STATE"].astype("string").str.upper().str.strip()
    drop(~df["worksite_state"].isin(US_STATES), "invalid_worksite_state")

    df["job_title"] = normalize_text(df["JOB_TITLE"])
    df["employer_name"] = normalize_text(df["EMPLOYER_NAME"])
    df["worksite_city"] = normalize_text(df["WORKSITE_CITY"])
    drop(df["job_title"].isna() | df["employer_name"].isna(), "missing_title_or_employer")

    df["full_time"] = df["FULL_TIME_POSITION"].astype("string").str.upper().str.strip().eq("Y")
    naics = df["NAICS_CODE"] if "NAICS_CODE" in df else pd.Series(pd.NA, index=df.index)
    df["naics_2"] = naics.astype("string").str.extract(r"^(\d{2})")[0]

    df["fiscal_year"] = fiscal_year(df["decision_date"])
    df = df.rename(columns={"CASE_NUMBER": "case_number"})[OUTPUT_COLUMNS].reset_index(drop=True)

    report["rows_out"] = len(df)
    report["rows_by_fiscal_year"] = {str(k): int(v) for k, v in df["fiscal_year"].value_counts().sort_index().items()}
    report["annual_wage_quantiles"] = {
        str(q): float(v) for q, v in df["annual_wage"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).items()
    }
    return df, report


def load_interim(interim_dir: Path) -> pd.DataFrame:
    files = sorted(interim_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {interim_dir}. Run `python -m h1b.ingest` first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    paths = get_paths(params)

    df, report = clean(load_interim(paths.interim), params["filters"])
    out = paths.processed / "lca_clean.parquet"
    df.to_parquet(out, index=False)
    (paths.reports / "ingest_report.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s (%d of %d rows kept)", out, report["rows_out"], report["rows_in"])
    log.info("dropped: %s", report["dropped"])


if __name__ == "__main__":
    main()
