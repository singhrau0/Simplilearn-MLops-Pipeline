"""Stage 5: Check processed train/test features and write JSON."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "artifacts/split"
PROC = ROOT / "artifacts/processed"
OUT = ROOT / "artifacts/reports/post_preprocessing_validation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(PROC / "train_features.csv")
test = pd.read_csv(PROC / "test_features.csv")
raw_train = pd.read_csv(SPLIT / "train.csv")
raw_test = pd.read_csv(SPLIT / "test.csv")
checks = {
    "same_feature_columns": train.columns.tolist() == test.columns.tolist(),
    "no_train_nulls": not train.isna().any().any(),
    "no_test_nulls": not test.isna().any().any(),
    "train_values_finite": bool(np.isfinite(train.to_numpy()).all()),
    "test_values_finite": bool(np.isfinite(test.to_numpy()).all()),
    "train_rows_match": len(train) == len(raw_train),
    "test_rows_match": len(test) == len(raw_test),
    "no_target_in_features": "Loan_Status" not in train.columns,
    "features_not_empty": train.shape[1] > 0
}
report = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "train_shape": list(train.shape),
    "test_shape": list(test.shape),
    "feature_columns": train.columns.tolist()
}
OUT.write_text(json.dumps(report, indent=4), encoding="utf-8")
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Post_Preprocessing_Validation"):
    mlflow.log_metric("checks_passed", sum(checks.values()))
    mlflow.log_metric("checks_total", len(checks))
    mlflow.set_tag("validation_status", report["status"])
    mlflow.log_artifact(str(OUT))
print("Post-preprocessing validation:", report["status"])
if report["status"] != "PASS":
    raise ValueError("Post-preprocessing checks failed; see JSON report")
