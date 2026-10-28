# Project Proposal

**Group:** Jiwon Choi, Sola Shin, Alex Bacro, Faith Kirabo

**Topic:** H-1B Salary Prediction

**Dataset:** We will use the U.S. Department of Labor OFLC Labor Condition Application (LCA) Disclosure Data (https://www.dol.gov/agencies/eta/foreign-labor/performance), a public dataset of H-1B employer filings released by fiscal year. Each record includes job title, SOC occupation code, employer, worksite location, offered wage, and wage unit. Multiple fiscal years provide enough data for training, validation, and an isolated test set.

**Problem statement:** We will predict the annualized offered salary (log-transformed regression) for an H-1B position from job title, SOC code, employer, location, and full-time status. This helps job platforms and candidates judge whether an offer is competitive. Prevailing-wage fields are excluded to avoid leakage. Success is measured by MAE, MAPE, and log-RMSE against a baseline of median wage by SOC and state. We train on earlier fiscal years, validate on a later year, and keep the latest year as an isolated test set until production validation.

**MLOps toolstack:** DOL Excel files → Parquet conversion (schema mapping, wage-unit normalization, validation) → Airflow DAG (feature engineering, training) → MLflow (FLAML AutoML, experiment tracking, Model Registry with semantic versioning) → Docker + FastAPI /predict endpoint → prediction logs → Evidently monitoring dashboard.

**Drift simulation:** Real drift comes from wage inflation and schema changes across fiscal years. Synthetic drift includes unconverted hourly wages, out-of-range values, and swapped state/city columns, with Evidently alerts documenting detection. GitHub Actions provides CI, and Docker Compose enables local reproduction.

