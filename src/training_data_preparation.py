"""Stage 6: Save final X_train and y_train files."""
from pathlib import Path
import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/final"
OUT.mkdir(parents=True, exist_ok=True)
X_train = pd.read_csv(ROOT / "artifacts/processed/train_features.csv")
train = pd.read_csv(ROOT / "artifacts/split/train.csv")

# Map loan approval labels to numbers: N=0, Y=1.
y_train = train["Loan_Status"].map({"N": 0, "Y": 1})
if y_train.isna().any():
    raise ValueError("Unexpected Loan_Status label in training data")
X_train.to_csv(OUT / "X_train.csv", index=False)
y_train.rename("Loan_Status").to_csv(OUT / "y_train.csv", index=False)
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Training_Data_Preparation"):
    mlflow.log_metric("train_rows", len(X_train))
    mlflow.log_metric("train_features", X_train.shape[1])
print("Saved X_train.csv and y_train.csv")
