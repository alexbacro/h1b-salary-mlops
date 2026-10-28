"""Canonical column schema for the OFLC LCA disclosure files.

DOL renames and re-suffixes columns between fiscal years (e.g. FY2019 uses
WORKSITE_CITY_1, FY2020+ uses WORKSITE_CITY; "H-1B_DEPENDENT" contains a hyphen).
Every raw header is passed through `normalize_header` and then mapped onto the
names below so downstream code only ever sees one schema.

Prevailing-wage fields (PREVAILING_WAGE, PW_*) are deliberately NOT ingested:
the prevailing wage is set from the same SOC/area/level as the offered wage and
would leak the target.
"""

from __future__ import annotations

import re

# Columns the pipeline cannot run without. Ingestion fails loudly if any is missing.
REQUIRED_COLUMNS = [
    "CASE_NUMBER",
    "CASE_STATUS",
    "DECISION_DATE",
    "VISA_CLASS",
    "JOB_TITLE",
    "SOC_CODE",
    "FULL_TIME_POSITION",
    "EMPLOYER_NAME",
    "WORKSITE_CITY",
    "WORKSITE_STATE",
    "WAGE_RATE_OF_PAY_FROM",
    "WAGE_UNIT_OF_PAY",
]

# Kept when present (useful for EDA / candidate features), silently skipped otherwise.
OPTIONAL_COLUMNS = [
    "RECEIVED_DATE",
    "BEGIN_DATE",
    "SOC_TITLE",
    "NAICS_CODE",
    "EMPLOYER_STATE",
    "WORKSITE_POSTAL_CODE",
    "WAGE_RATE_OF_PAY_TO",
    "TOTAL_WORKER_POSITIONS",
    "NEW_EMPLOYMENT",
    "CONTINUED_EMPLOYMENT",
    "H_1B_DEPENDENT",
]

CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Explicit renames for headers that differ by more than formatting.
ALIASES = {
    "H1B_DEPENDENT": "H_1B_DEPENDENT",
    "EMPLOYER_BUSINESS_NAME": "EMPLOYER_NAME",
}


def normalize_header(name: object) -> str:
    """Upper-case and replace any run of non-alphanumerics with '_' ("H-1B Dependent" -> "H_1B_DEPENDENT")."""
    return re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_")


def canonical_name(name: object) -> str | None:
    """Map a raw header to its canonical column, or None if we don't ingest it.

    Handles the FY2019-style "_1" suffix (first worksite) by stripping it.
    """
    norm = normalize_header(name)
    norm = ALIASES.get(norm, norm)
    if norm in CANONICAL_COLUMNS:
        return norm
    if norm.endswith("_1") and norm[:-2] in CANONICAL_COLUMNS:
        return norm[:-2]
    return None
