"""Stage 1: Simple EDA report for cleaned loan data."""
import json
from pathlib import Path
import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts/cleaned/loan_cleaned.csv"
OUT = ROOT / "artifacts/reports/eda.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)

# Describe the data without modifying it.
report = {
    "rows": len(df), "columns": len(df.columns),
    "column_names": df.columns.tolist(),
    "data_types": df.dtypes.astype(str).to_dict(),
    "missing_by_column": df.isna().sum().astype(int).to_dict(),
    "duplicates": int(df.duplicated().sum()),
    "target_distribution": df["Loan_Status"].value_counts(dropna=False).to_dict(),
    "numeric_summary": df.describe().round(2).fillna(0).to_dict(),
    "categorical_summary": {
        col: df[col].value_counts(dropna=False).astype(int).to_dict()
        for col in df.select_dtypes(include="object").columns
    }
}
OUT.write_text(json.dumps(report, indent=4, default=str), encoding="utf-8")
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Data_Analysis_EDA"):
    mlflow.log_metric("rows", len(df))
    mlflow.log_metric("columns", len(df.columns))
    mlflow.log_artifact(str(OUT))
print("EDA JSON saved:", OUT)
