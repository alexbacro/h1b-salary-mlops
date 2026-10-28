import pandas as pd
import pytest

from h1b.clean import clean, fiscal_year, normalize_soc, normalize_text, parse_wage


def test_clean_applies_every_rule(raw_df, filters):
    df, report = clean(raw_df, filters)

    assert sorted(df["case_number"]) == ["2", "3", "6", "I-200-24001-000001"]
    assert report["rows_in"] == 11
    assert report["rows_out"] == 4
    assert report["dropped"] == {
        "missing_decision_date": 0,
        "duplicate_case_number": 1,
        "visa_class_out_of_scope": 1,
        "case_status_not_certified": 1,
        "unknown_wage_unit": 1,
        "unparseable_wage": 0,
        "annual_wage_out_of_range": 1,
        "invalid_soc_code": 1,
        "invalid_worksite_state": 1,
        "missing_title_or_employer": 0,
    }


def test_clean_annualizes_and_keeps_latest_duplicate(raw_df, filters):
    df, _ = clean(raw_df, filters)
    wages = df.set_index("case_number")["annual_wage"]
    assert wages["I-200-24001-000001"] == 125000  # later duplicate won
    assert wages["2"] == 50 * 2080
    assert wages["3"] == 4000 * 26


def test_clean_output_features(raw_df, filters):
    df, _ = clean(raw_df, filters)
    row = df.set_index("case_number").loc["I-200-24001-000001"]
    assert row["soc_code"] == "15-1252"
    assert row["soc_major"] == "15"
    assert row["job_title"] == "SOFTWARE ENGINEER II"
    assert row["employer_name"] == "ACME CORP"
    assert row["full_time"]
    assert row["naics_2"] == "54"
    assert row["fiscal_year"] == 2025
    assert row["log_wage"] == pytest.approx(11.736, abs=1e-3)


def test_clean_output_has_no_prevailing_wage_columns(raw_df, filters):
    df, _ = clean(raw_df, filters)
    assert not any("PW" in c.upper() or "PREVAILING" in c.upper() for c in df.columns)


def test_normalize_soc():
    s = pd.Series(["15-1252.00", "151252", "15-1252", "1512", None])
    assert normalize_soc(s).tolist()[:3] == ["15-1252"] * 3
    assert normalize_soc(s).isna().tolist()[3:] == [True, True]


def test_parse_wage():
    s = pd.Series(["$85,000.00", "85000 - 95000", "45.5", "", None])
    out = parse_wage(s)
    assert out.tolist()[:3] == [85000.0, 85000.0, 45.5]
    assert out.isna().tolist()[3:] == [True, True]


def test_normalize_text_blanks_to_na():
    out = normalize_text(pd.Series(["  data   scientist ", "!!!"]))
    assert out[0] == "DATA SCIENTIST"
    assert pd.isna(out[1])


def test_fiscal_year_boundary():
    dates = pd.to_datetime(pd.Series(["2024-09-30", "2024-10-01"]))
    assert fiscal_year(dates).tolist() == [2024, 2025]
