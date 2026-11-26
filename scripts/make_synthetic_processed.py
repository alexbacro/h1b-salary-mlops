"""Write a tiny synthetic data/processed (train.parquet + val.parquet) for the Docker smoke test in CI.

CI has no DOL data, so this builds a handful of raw filings, runs them through the real
`h1b.clean.clean()` (same schema and normalization as `make data`) and splits them on the fiscal
years in params.yaml. The (15-1252, CA) training rows are 140k / 150k / 160k, so the baseline
champion must predict exactly 150,000 for that pair; the smoke test checks for that value.

Usage:
    python scripts/make_synthetic_processed.py OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from h1b.clean import clean
from h1b.config import load_params

EXPECTED_CA_SOFTWARE_MEDIAN = 150_000

# (decision date, SOC code as filed, state, annual wage). FY = Oct-Sep, so 2022-03 is FY2022.
ROWS = [
    ("2022-03-01", "15-1132.00", "CA", 140_000),  # 2010 code, remapped to 15-1252 by clean()
    ("2023-03-01", "15-1252.00", "CA", 150_000),
    ("2023-04-01", "15-1252.00", "CA", 160_000),
    ("2022-05-01", "15-1252.00", "TX", 110_000),
    ("2023-05-01", "13-2051.00", "NY", 95_000),
    ("2024-03-01", "15-1252.00", "CA", 165_000),  # val
    ("2024-04-01", "15-1252.00", "TX", 115_000),  # val
    ("2024-05-01", "13-2051.00", "NY", 99_000),  # val
]


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CASE_NUMBER": f"I-200-{i:05d}",
            "CASE_STATUS": "Certified",
            "DECISION_DATE": date,
            "VISA_CLASS": "H-1B",
            "JOB_TITLE": "Software Engineer",
            "SOC_CODE": soc,
            "FULL_TIME_POSITION": "Y",
            "EMPLOYER_NAME": "Acme Corp",
            "WORKSITE_CITY": "Springfield",
            "WORKSITE_STATE": state,
            "WAGE_RATE_OF_PAY_FROM": str(wage),
            "WAGE_UNIT_OF_PAY": "Year",
            "NAICS_CODE": "541511",
        }
        for i, (date, soc, state, wage) in enumerate(ROWS)
    )


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    params = load_params()
    df, _ = clean(raw_frame(), params["filters"])
    for name, years in [("train", params["split"]["train_years"]), ("val", params["split"]["val_years"])]:
        part = df[df["fiscal_year"].isin(years)]
        part.to_parquet(out / f"{name}.parquet", index=False)
        print(f"wrote {out / f'{name}.parquet'} ({len(part)} rows)")


if __name__ == "__main__":
    main()
