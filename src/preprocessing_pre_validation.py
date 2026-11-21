"""Stage 2: Check whether cleaned data is ready for splitting."""
import json
from pathlib import Path
import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts/cleaned/loan_cleaned.csv"
OUT = ROOT / "artifacts/reports/preprocessing_pre_validation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)

# Loan_ID is an identifier, not a predictive feature.
features = df.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
num_cols = features.select_dtypes(include="number").columns.tolist()
cat_cols = features.select_dtypes(exclude="number").columns.tolist()
checks = {
    "target_exists": "Loan_Status" in df.columns,
    "target_has_no_nulls": "Loan_Status" in df and df["Loan_Status"].notna().all(),
    "target_is_binary": "Loan_Status" in df and df["Loan_Status"].isin(["Y", "N"]).all(),
    "has_features": len(features.columns) > 0,
    "has_enough_rows": len(df) >= 10
}
report = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": {k: bool(v) for k, v in checks.items()},
    "numerical_features": num_cols,
    "categorical_features": cat_cols,
    "missing_by_column": features.isna().sum().astype(int).to_dict()
}
OUT.write_text(json.dumps(report, indent=4), encoding="utf-8")
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Preprocessing_Pre_Validation"):
    mlflow.log_metric("check_passed", int(report["status"] == "PASS"))
    mlflow.log_artifact(str(OUT))
print("Preprocessing pre-validation:", report["status"])
if report["status"] != "PASS":
    raise ValueError("Preprocessing validation failed; see JSON report")
