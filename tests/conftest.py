"""Shared fixtures: a tiny raw LCA frame that exercises every cleaning rule."""

import pandas as pd
import pytest


def raw_row(**overrides) -> dict:
    """One valid raw record in canonical-column form; override fields per test."""
    row = {
        "CASE_NUMBER": "I-200-24001-000001",
        "CASE_STATUS": "Certified",
        "DECISION_DATE": "2024-11-15",
        "VISA_CLASS": "H-1B",
        "JOB_TITLE": "Software Engineer, II",
        "SOC_CODE": "15-1252.00",
        "FULL_TIME_POSITION": "Y",
        "EMPLOYER_NAME": "Acme Corp.",
        "WORKSITE_CITY": "Chicago",
        "WORKSITE_STATE": "IL",
        "WAGE_RATE_OF_PAY_FROM": "120000",
        "WAGE_UNIT_OF_PAY": "Year",
        "NAICS_CODE": "541511",
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rows = [
        raw_row(),  # valid, FY2025
        raw_row(CASE_NUMBER="2", WAGE_RATE_OF_PAY_FROM="$50.00", WAGE_UNIT_OF_PAY="Hour"),  # 104000 / yr
        raw_row(CASE_NUMBER="3", WAGE_RATE_OF_PAY_FROM="4,000", WAGE_UNIT_OF_PAY="Bi-Weekly"),  # 104000 / yr
        raw_row(CASE_NUMBER="4", VISA_CLASS="E-3 Australian"),  # out of scope
        raw_row(CASE_NUMBER="5", CASE_STATUS="Denied"),  # not certified
        raw_row(CASE_NUMBER="6", CASE_STATUS="Certified - Withdrawn", DECISION_DATE="2023-09-30"),  # FY2023
        raw_row(CASE_NUMBER="7", WAGE_RATE_OF_PAY_FROM="45", WAGE_UNIT_OF_PAY="Year"),  # unconverted hourly
        raw_row(CASE_NUMBER="8", WAGE_UNIT_OF_PAY="Fortnight"),  # unknown unit
        raw_row(CASE_NUMBER="9", SOC_CODE="garbage"),  # invalid SOC
        raw_row(CASE_NUMBER="10", WORKSITE_STATE="Chicago"),  # swapped city/state
        # duplicate of case 1 with a later decision: should replace it
        raw_row(DECISION_DATE="2024-12-01", WAGE_RATE_OF_PAY_FROM="125000"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def filters() -> dict:
    return {
        "visa_classes": ["H-1B"],
        "case_statuses": ["CERTIFIED", "CERTIFIED-WITHDRAWN"],
        "min_annual_wage": 15000,
        "max_annual_wage": 1000000,
    }
