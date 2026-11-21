
from pathlib import Path
import pandas as pd
import mlflow
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "artifacts" / "final"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
OUTPUT_DIR = BASE_DIR / "artifacts" / "testing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load testing datasets
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Testing datasets loaded successfully!")
print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)


# 3. Define saved model file names
models = [
    "Logistic_Regression",
    "KNN",
    "Decision_Tree"
]


# 4. Start MLflow experiment
mlflow.set_experiment("Loan_Data_Pipeline")

all_metrics = {}


# 5. Test each model
with mlflow.start_run(run_name="Model_Testing"):

    for name in models:

        print("\nTesting Model:", name)

        # Load trained model
        model_path = MODEL_DIR / (name + ".joblib")
        model = joblib.load(model_path)

        # Predict using testing dataset
        y_pred = model.predict(X_test)

        # 6. Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        # Display metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # 7. Store metrics
        all_metrics[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # 8. Log metrics to MLflow
        mlflow.log_metric(name + "_accuracy", accuracy)
        mlflow.log_metric(name + "_precision", precision)
        mlflow.log_metric(name + "_recall", recall)
        mlflow.log_metric(name + "_f1_score", f1)

        # 9. Generate confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            cmap="Blues"
        )

        plt.title(name + " - Confusion Matrix")

        # Save confusion matrix image
        cm_path = OUTPUT_DIR / (name + "_confusion_matrix.png")

        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        print("Confusion Matrix Saved:", cm_path)

        # Log confusion matrix to MLflow
        mlflow.log_artifact(str(cm_path))


    # 10. Save all testing metrics as JSON
    metrics_path = OUTPUT_DIR / "testing_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(all_metrics, file, indent=4)

    # 11. Save metrics as CSV for comparison
    metrics_df = pd.DataFrame(all_metrics).T
    csv_path = OUTPUT_DIR / "testing_metrics.csv"

    metrics_df.to_csv(csv_path, index_label="model")

    # 12. Log testing reports to MLflow
    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(str(csv_path))


# 13. Final message
print("\nAll models tested successfully!")
print("Testing metrics saved:", metrics_path)
print("CSV report saved:", csv_path)
