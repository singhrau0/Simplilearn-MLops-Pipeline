
from pathlib import Path

import pandas as pd
import mlflow


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "loan_train.csv"
OUTPUT_PATH = BASE_DIR / "artifacts" / "cleaned" / "loan_cleaned.csv"


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original Shape:", df.shape)

# Store original statistics
original_rows = len(df)
missing_before = int(df.isnull().sum().sum())


# 3. Start MLflow tracking
mlflow.set_experiment("Loan_Data_Pipeline")

with mlflow.start_run(run_name="Data_Cleaning"):

    # 4. Remove duplicate rows
    df = df.drop_duplicates()

    # 5. Remove rows with missing target values
    df = df.dropna(subset=["Loan_Status"])

    # 6. Fill missing numerical values with median
    numerical_columns = [
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # 7. Fill missing categorical values with mode
    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed"
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # 8. Calculate cleaning statistics
    cleaned_rows = len(df)
    missing_after = int(df.isnull().sum().sum())
    removed_rows = original_rows - cleaned_rows

    print("\nCleaning Summary:")
    print("Original Rows:", original_rows)
    print("Cleaned Rows:", cleaned_rows)
    print("Removed Rows:", removed_rows)
    print("Missing Before:", missing_before)
    print("Missing After:", missing_after)

    # 9. Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset saved:", OUTPUT_PATH)

    # 10. Log metrics to MLflow
    mlflow.log_metric("original_rows", original_rows)
    mlflow.log_metric("cleaned_rows", cleaned_rows)
    mlflow.log_metric("removed_rows", removed_rows)
    mlflow.log_metric("missing_before", missing_before)
    mlflow.log_metric("missing_after", missing_after)

    # 11. Save cleaned dataset as MLflow artifact
    mlflow.log_artifact(str(OUTPUT_PATH))

print("\nData cleaning completed successfully!")
