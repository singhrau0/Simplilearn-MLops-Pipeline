
import subprocess
import sys
from pathlib import Path

import mlflow


# 1. Define project paths
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent


# 2. Set MLflow experiment name
mlflow.set_experiment("Loan_Data_Pipeline")


# 3. Define pipeline stages
stages = [
    "pre_cleaning_validation.py",
    "data_cleaning.py",
    "post_cleaning_validation.py",
    "data_analysis.py",
    "preprocessing_pre_validation.py",
    "data_splitting.py",
    "data_preprocessing.py",
    "post_preprocessing_validation.py",
    "training_data_preparation.py",
    "testing_data_preparation.py",
    "model_training.py",
    "model_testing.py",
    "select_champion.py"

]


# 4. Execute pipeline stages one by one
for stage in stages:

    print(f"\n========== Starting {stage} ==========\n")

    subprocess.run(
        [sys.executable, str(SRC_DIR / stage)],
        cwd=BASE_DIR,
        check=True
    )

    print(f"\n========== Completed {stage} ==========\n")


# 5. Pipeline completed
print("\nMLOps Pipeline Completed Successfully!")
print("MLflow Experiment: Loan_Data_Pipeline")
