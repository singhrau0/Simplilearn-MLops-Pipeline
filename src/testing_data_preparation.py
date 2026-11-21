"""Stage 7: Save final X_test and y_test files."""
from pathlib import Path
import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/final"
OUT.mkdir(parents=True, exist_ok=True)
X_test = pd.read_csv(ROOT / "artifacts/processed/test_features.csv")
test = pd.read_csv(ROOT / "artifacts/split/test.csv")

# Use the same target mapping as training.
y_test = test["Loan_Status"].map({"N": 0, "Y": 1})
if y_test.isna().any():
    raise ValueError("Unexpected Loan_Status label in testing data")
X_test.to_csv(OUT / "X_test.csv", index=False)
y_test.rename("Loan_Status").to_csv(OUT / "y_test.csv", index=False)
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Testing_Data_Preparation"):
    mlflow.log_metric("test_rows", len(X_test))
    mlflow.log_metric("test_features", X_test.shape[1])
print("Saved X_test.csv and y_test.csv")
