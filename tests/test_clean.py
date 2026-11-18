import pandas as pd
import pytest

from h1b.clean import clean, fiscal_year, normalize_soc, normalize_text, parse_wage, to_soc_2018
from h1b.soc_crosswalk import ONET_OVERRIDES, SOC_2010_TO_2018

from .conftest import raw_row


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
    s = pd.Series(["15-1252.00", "151252", "15-1252", "15-1252.00.00", "1512", None])
    assert normalize_soc(s).tolist()[:4] == ["15-1252"] * 4
    assert normalize_soc(s).isna().tolist()[4:] == [True, True]


def test_to_soc_2018_remaps_retired_2010_codes():
    s = pd.Series(["15-1132.00", "15-1133", "15-1121.00", "15-1199.00", "15-1134.00"])
    assert to_soc_2018(s).tolist() == ["15-1252", "15-1252", "15-1211", "15-1299", "15-1254"]


def test_to_soc_2018_uses_onet_detail_when_it_names_the_occupation():
    s = pd.Series(["15-1199.01", "15-1199.06", "29-1069.04", "15-1199.08", "151199.08", "15-1199.00"])
    assert to_soc_2018(s).tolist() == ["15-1253", "15-1243", "29-1217", "15-2051", "15-2051", "15-1299"]


def test_to_soc_2018_overrides_detail_codes_under_parents_still_valid_in_2018():
    # 17-2199 still exists in 2018, but O*NET moved Validation Engineers (.02) to Industrial Engineers.
    s = pd.Series(["17-2199.02", "17-2199.00", "13-2051.00"])
    assert to_soc_2018(s).tolist() == ["17-2112", "17-2199", "13-2051"]


def test_to_soc_2018_keeps_2018_codes_and_invalid_as_na():
    s = pd.Series(["15-1252.00", "13-1199.00", "garbage", None])
    out = to_soc_2018(s)
    assert out.tolist()[:2] == ["15-1252", "13-1199"]  # 13-1199 is valid in both versions
    assert out.isna().tolist()[2:] == [True, True]


def test_crosswalk_targets_are_never_retired_codes():
    targets = set(SOC_2010_TO_2018.values()) | set(ONET_OVERRIDES.values())
    assert not targets & set(SOC_2010_TO_2018)  # one pass is enough; no chains


def test_clean_reports_remapped_soc_codes(filters):
    raw = pd.DataFrame([raw_row(), raw_row(CASE_NUMBER="2", SOC_CODE="15-1132.00")])
    df, report = clean(raw, filters)
    assert df.set_index("case_number").loc["2", "soc_code"] == "15-1252"
    assert df.set_index("case_number").loc["2", "soc_major"] == "15"
    assert report["soc_2010_remapped"] == 1


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
