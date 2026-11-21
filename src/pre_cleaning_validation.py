
import sys
import json
import subprocess
from pathlib import Path

import pandas as pd
import mlflow


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_train.csv"
REPORT_PATH = BASE_DIR / "artifacts" / "pre_validation.json"


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# 3. Start MLflow tracking
mlflow.set_experiment("Loan_Data_Pipeline")

with mlflow.start_run(run_name="Pre_Cleaning_Validation"):

    # 4. Check missing values
    missing_values = df.isnull().sum()

    print("\nMissing Values:")
    print(missing_values)

    # 5. Check duplicate rows
    duplicates = int(df.duplicated().sum())
    print("\nDuplicate Rows:", duplicates)

    # 6. Check column data types
    print("\nColumn Data Types:")
    print(df.dtypes)

    # 7. Check target column
    target_missing = int(df["Loan_Status"].isnull().sum())
    print("\nMissing Loan_Status:", target_missing)

    # 8. Save metrics to MLflow
    mlflow.log_metric("total_rows", len(df))
    mlflow.log_metric("total_columns", len(df.columns))
    mlflow.log_metric("missing_values", int(missing_values.sum()))
    mlflow.log_metric("duplicate_rows", duplicates)
    mlflow.log_metric("target_missing", target_missing)

    # 9. Create validation report
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing_values.to_dict(),
        "duplicate_rows": duplicates,
        "target_missing": target_missing,
        "data_types": df.dtypes.astype(str).to_dict()
    }

    # 10. Save report as JSON
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    mlflow.log_artifact(str(REPORT_PATH))

    print("\nValidation report saved!")

    # 11. DVC versioning of raw dataset
    print("\nRunning DVC versioning...")

    subprocess.run(
        [sys.executable, "-m", "dvc", "add", "data/loan_train.csv"],
        cwd=BASE_DIR,
        check=True
    )

    print("DVC dataset tracking completed!")

print("\nPre-cleaning validation completed!")
