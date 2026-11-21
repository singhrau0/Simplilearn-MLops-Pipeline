
from pathlib import Path
import json
import joblib
import mlflow


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_PATH = BASE_DIR / "artifacts" / "testing" / "testing_metrics.json"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
OUTPUT_DIR = BASE_DIR / "artifacts" / "champion"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load testing metrics
with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

print("Testing metrics loaded successfully!")


# 3. Select model with highest precision
champion_name = max(
    metrics,
    key=lambda name: metrics[name]["precision"]
)

champion_precision = metrics[champion_name]["precision"]

print("\nChampion Model:", champion_name)
print("Champion Precision:", champion_precision)


# 4. Load selected model
model_path = MODEL_DIR / (champion_name + ".joblib")

champion_model = joblib.load(model_path)


# 5. Save champion model
champion_path = OUTPUT_DIR / "champion.joblib"

joblib.dump(champion_model, champion_path)

print("Champion Model Saved:", champion_path)


# 6. Save champion model information
champion_info = {
    "champion_model": champion_name,
    "precision": champion_precision
}

info_path = OUTPUT_DIR / "champion_info.json"

with open(info_path, "w") as file:
    json.dump(champion_info, file, indent=4)


# 7. Log champion model details in MLflow
mlflow.set_experiment("Loan_Data_Pipeline")

with mlflow.start_run(run_name="Champion_Selection"):

    mlflow.log_param("champion_model", champion_name)
    mlflow.log_metric("champion_precision", champion_precision)

    mlflow.log_artifact(str(champion_path))
    mlflow.log_artifact(str(info_path))


print("\nChampion model selection completed successfully!")
