"""Stage 3: Split cleaned data into train and test rows."""
from pathlib import Path
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts/cleaned/loan_cleaned.csv"
OUT = ROOT / "artifacts/split"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)

# Stratification preserves the loan approval ratio in both sets.
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["Loan_Status"]
)
train_df.to_csv(OUT / "train.csv", index=False)
test_df.to_csv(OUT / "test.csv", index=False)
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Data_Splitting"):
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))
print("Saved train.csv and test.csv:", len(train_df), len(test_df))
