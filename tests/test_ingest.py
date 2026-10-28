import pandas as pd
import pytest

from h1b.ingest import SchemaError, convert_all, read_lca_excel
from h1b.schema import canonical_name

from .conftest import raw_row


def test_canonical_name_handles_year_to_year_variants():
    assert canonical_name("H-1B_DEPENDENT") == "H_1B_DEPENDENT"
    assert canonical_name("worksite_city_1") == "WORKSITE_CITY"  # FY2019 layout
    assert canonical_name(" Wage Rate Of Pay From ") == "WAGE_RATE_OF_PAY_FROM"
    assert canonical_name("PREVAILING_WAGE") is None  # leakage: never ingested
    assert canonical_name("EMPLOYER_POC_EMAIL") is None


def test_read_lca_excel_maps_schema_and_drops_extra_columns(tmp_path):
    row = raw_row()
    # Rename a few headers to mimic an older layout, and add columns we must not keep.
    row["WORKSITE_STATE_1"] = row.pop("WORKSITE_STATE")
    row["PREVAILING_WAGE"] = "110000"
    row["EMPLOYER_POC_EMAIL"] = "hr@acme.com"
    path = tmp_path / "LCA_Disclosure_Data_FY2025_Q1.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False)

    df = read_lca_excel(path)
    assert "WORKSITE_STATE" in df.columns
    assert "PREVAILING_WAGE" not in df.columns
    assert "EMPLOYER_POC_EMAIL" not in df.columns
    assert df.loc[0, "SOURCE_FILE"] == path.name
    assert df.loc[0, "WAGE_RATE_OF_PAY_FROM"] == "120000"  # read as string


def test_read_lca_excel_fails_on_missing_required_column(tmp_path):
    row = raw_row()
    del row["SOC_CODE"]
    path = tmp_path / "bad.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False)
    with pytest.raises(SchemaError, match="SOC_CODE"):
        read_lca_excel(path)


def test_convert_all_writes_and_caches(tmp_path, caplog):
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    pd.DataFrame([raw_row()]).to_excel(raw / "a.xlsx", index=False)

    outputs = convert_all(raw, interim)
    assert [p.name for p in outputs] == ["a.parquet"]
    assert len(pd.read_parquet(outputs[0])) == 1

    caplog.set_level("INFO")
    convert_all(raw, interim)
    assert "cached" in caplog.text


def test_convert_all_errors_when_no_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        convert_all(tmp_path, tmp_path)
