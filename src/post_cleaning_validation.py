
import json
from pathlib import Path

import pandas as pd
import mlflow


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "loan_train.csv"
CLEAN_PATH = BASE_DIR / "artifacts" / "cleaned" / "loan_cleaned.csv"
REPORT_PATH = BASE_DIR / "artifacts" / "post_validation.json"


# 2. Load datasets
raw_df = pd.read_csv(RAW_PATH)
df = pd.read_csv(CLEAN_PATH)

print("Original Dataset:", raw_df.shape)
print("Cleaned Dataset:", df.shape)


# 3. Calculate data quality metrics
missing_before = int(raw_df.isnull().sum().sum())
missing_after = int(df.isnull().sum().sum())

duplicates_before = int(raw_df.duplicated().sum())
duplicates_after = int(df.duplicated().sum())


# 4. Perform post-cleaning validation
checks = {
    "no_missing_values": missing_after == 0,
    "no_duplicate_rows": duplicates_after == 0,
    "valid_loan_status": bool(df["Loan_Status"].isin(["Y", "N"]).all()),
    "valid_income": bool(
        ((df["ApplicantIncome"] >= 0) &
         (df["CoapplicantIncome"] >= 0)).all()
    ),
    "valid_loan_amount": bool((df["LoanAmount"] > 0).all()),
    "valid_credit_history": bool(
        df["Credit_History"].isin([0, 1]).all()
    ),
    "unique_loan_ids": bool(df["Loan_ID"].is_unique),
    "same_columns": set(raw_df.columns) == set(df.columns),
    "dataset_not_empty": len(df) > 0
}

overall_status = "PASS" if all(checks.values()) else "FAIL"


# 5. Create JSON report
report = {
    "overall_status": overall_status,
    "before_cleaning": {
        "rows": len(raw_df),
        "columns": len(raw_df.columns),
        "missing_values": missing_before,
        "duplicate_rows": duplicates_before
    },
    "after_cleaning": {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_after,
        "duplicate_rows": duplicates_after
    },
    "validation_checks": checks,
    "missing_values_by_column": df.isnull().sum().to_dict()
}


# 6. Save JSON report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w") as file:
    json.dump(report, file, indent=4)


# 7. Track results using MLflow
mlflow.set_experiment("Loan_Data_Pipeline")

with mlflow.start_run(run_name="Post_Cleaning_Validation"):

    mlflow.log_metric("missing_before", missing_before)
    mlflow.log_metric("missing_after", missing_after)
    mlflow.log_metric("duplicates_before", duplicates_before)
    mlflow.log_metric("duplicates_after", duplicates_after)

    for name, passed in checks.items():
        mlflow.log_metric(name, int(passed))

    mlflow.set_tag("validation_status", overall_status)

    mlflow.log_artifact(str(REPORT_PATH))


# 8. Print final result
print("\nPost-Cleaning Validation:", overall_status)
print("JSON Report Saved:", REPORT_PATH)
