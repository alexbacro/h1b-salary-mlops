# Data

Source: U.S. Department of Labor, Office of Foreign Labor Certification (OFLC),
**LCA Disclosure Data** (Form ETA-9035: H-1B, H-1B1, E-3), from the
[performance data page](https://www.dol.gov/agencies/eta/foreign-labor/performance).
Each row is one Labor Condition Application, which an employer files before
sponsoring an H-1B worker. It records the job, the worksite, and the wage offered.

## What to download

DOL blocks scripted downloads, so **download these in a browser** and save them
unchanged into `data/raw/`. Each file is roughly 100 MB.

| Split (see `params.yaml`) | Fiscal year | Files |
|---|---|---|
| train | FY2022 | [Q1](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2022_Q1.xlsx) · [Q2](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2022_Q2.xlsx) · [Q3](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2022_Q3.xlsx) · [Q4](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2022_Q4.xlsx) |
| train | FY2023 | [Q1](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2023_Q1.xlsx) · [Q2](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2023_Q2.xlsx) · [Q3](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2023_Q3.xlsx) · [Q4](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2023_Q4.xlsx) |
| val | FY2024 | [Q1](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2024_Q1.xlsx) · [Q2](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2024_Q2.xlsx) · [Q3](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2024_Q3.xlsx) · [Q4](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2024_Q4.xlsx) |
| test (holdout) | FY2025 | [Q1](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2025_Q1.xlsx) · [Q2](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2025_Q2.xlsx) · [Q3](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2025_Q3.xlsx) · [Q4](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2025_Q4.xlsx) |

You don't need the "Worksites" or "Appendix A" files. Field definitions are in the
[FY2025 record layout](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Record_Layout_FY2025_Q4.pdf).

**To iterate quickly** you can start with one quarter per year (e.g. all four
Q1 files). The pipeline only needs at least one file per fiscal year listed in `params.yaml`.

### Why these years

- **FY2020 onward share one schema.** The disclosure layout was redesigned in FY2020
  (the files were renamed from "H-1B Disclosure Data" to "LCA Disclosure Data"), and
  FY2019 and earlier use different column names. The schema mapper
  (`src/h1b/schema.py`) handles the common FY2019 variants, but FY2022+ keeps
  training data on one consistent form.
- **FY2020–FY2021 are skipped** to keep COVID-era hiring out of training.
- **FY2025 is the latest complete year**, so it is the isolated test set. FY2026 is
  only partially published (Q3) and can serve later as "future" production traffic
  for real (non-synthetic) drift.

## Pipeline stages

| Stage | Command | Output |
|---|---|---|
| Excel → Parquet | `python -m h1b.ingest` | `data/interim/<file>.parquet` (canonical columns, all strings) |
| Clean + validate | `python -m h1b.clean` | `data/processed/lca_clean.parquet`, `reports/ingest_report.json` |
| Temporal split | `python -m h1b.split` | `data/processed/{train,val}.parquet`, `data/holdout/test.parquet` |
| Baseline | `python -m h1b.baseline` | `reports/baseline_metrics_val.json` |

## Cleaning rules (`src/h1b/clean.py`)

Applied in order. `reports/ingest_report.json` records how many rows each rule removed.

1. Drop rows with no `DECISION_DATE`; dedupe on `CASE_NUMBER`, keeping the latest decision.
2. Keep `VISA_CLASS == H-1B` and `CASE_STATUS` in {Certified, Certified-Withdrawn}.
3. Annualize `WAGE_RATE_OF_PAY_FROM` by `WAGE_UNIT_OF_PAY`: Hour × 2080, Week × 52,
   Bi-Weekly × 26, Month × 12, Year × 1. Drop unknown units and unparseable wages.
4. Drop annual wages outside [$15k, $1M]. This catches e.g. an hourly rate entered with unit "Year".
5. Normalize SOC to `NN-NNNN`; drop invalid codes. Retired 2010 SOC codes are moved onto 2018 codes
   (see [SOC versions](#soc-versions) below).
6. Drop worksite states that aren't US state/territory codes (catches swapped city/state).
7. Normalize job title, employer, and city text (upper-case, no punctuation); drop rows missing title or employer.

### SOC versions

The 2018 SOC revision renumbered many occupations, and the LCA files switched over during our
training years: **FY2022 mostly uses 2010 codes, FY2023 onward uses 2018 codes**. Before this
fix, 56% of FY2022 rows were on retired 2010 codes (software developers alone were 27.6% of FY2022
as `15-1132` versus 1.0% in FY2023, where they appear as `15-1252`). The same job was two
unrelated categories to the model, and the drift report flagged `soc_code` (Jensen-Shannon 0.32).

`clean.py` (`to_soc_2018`) now puts every row on 2018 codes, using the
[BLS 2010 → 2018 crosswalk](https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx):

- Only the 88 *retired* 2010 codes are remapped. Codes that still exist in 2018 are unchanged.
- 24 retired codes were split by BLS into several 2018 codes. Those go to the 2018 code with the
  same title if there is one (`15-1134` Web Developers → `15-1254` Web Developers), otherwise to
  the code most common in FY2023 filings (`15-1132` → `15-1252` Software Developers,
  `15-1199` → `15-1299` Computer Occupations, All Other).
- When the filing has an O*NET detail code that names a 2018 occupation, that wins
  (`15-1199.01` Software QA Engineers and Testers → `15-1253`). Seven such overrides exist.

The lookup lives in `src/h1b/soc_crosswalk.py`, generated by `scripts/build_soc_crosswalk.py`
(rules documented there). `reports/ingest_report.json` records how many rows were remapped
(`soc_2010_remapped`: 339,582 on FY2022–FY2025).

Effect on FY2022–FY2025: the code mismatch is gone (`15-1252` is 39% of FY2022 and 35% of FY2023),
`soc_code` drift train → val drops from 0.32 to 0.08 (below the 0.1 threshold), and no val SOC
code is unseen in train. Fixing a parsing bug at the same time (codes like `15-1132.00.00` were
rejected) keeps 2,020 more rows. The median baseline gets slightly *worse* (val MAE $27,254 →
$27,620): FY2022 wages are lower, and before the fix most FY2022 tech rows sat on codes val never
uses, so the baseline was effectively using FY2023 medians. Training on lower FY2022 wages without
a time trend biases predictions down; the model should account for wage growth (e.g. `fiscal_year`
as a feature or an inflation adjustment).

## Clean dataset columns

| Column | Type | Description |
|---|---|---|
| `case_number` | str | OFLC case ID (unique) |
| `decision_date` | datetime | Date of OFLC decision |
| `fiscal_year` | int | Federal FY of the decision (Oct–Sep); drives the split |
| `job_title` | str | Normalized job title (feature) |
| `soc_code` | str | 2018 SOC occupation code `NN-NNNN` (feature) |
| `soc_major` | str | SOC major group, first 2 digits (feature) |
| `employer_name` | str | Normalized employer legal name (feature) |
| `worksite_city` | str | Normalized first-worksite city (feature) |
| `worksite_state` | str | 2-letter first-worksite state (feature) |
| `full_time` | bool | Full-time position (feature) |
| `naics_2` | str | Employer industry sector, 2-digit NAICS (candidate feature) |
| `annual_wage` | float | Offered wage, annualized (USD) |
| `log_wage` | float | `ln(annual_wage)`: **regression target** |

### Excluded on purpose (leakage)

`PREVAILING_WAGE`, `PW_UNIT_OF_PAY`, `PW_WAGE_LEVEL`, and the other `PW_*` fields are never
ingested. The prevailing wage is determined from the same occupation, area, and level as
the offered wage, and employers must pay at least that amount, so including it would leak the target.
