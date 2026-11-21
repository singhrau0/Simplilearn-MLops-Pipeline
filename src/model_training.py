
from pathlib import Path
import pandas as pd
import mlflow
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "artifacts" / "final"
OUTPUT_DIR = BASE_DIR / "artifacts" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load training datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

print("Training datasets loaded successfully!")
print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)


# 3. Define machine learning models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision_Tree": DecisionTreeClassifier(random_state=42)
}


# 4. Define hyperparameters for each model
parameters = {
    "Logistic_Regression": {
        "C": [0.1, 1, 10]
    },

    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },

    "Decision_Tree": {
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10]
    }
}


# 5. Start MLflow experiment
mlflow.set_experiment("Loan_Data_Pipeline")

all_best_params = {}


# 6. Train and tune all three models
with mlflow.start_run(run_name="Model_Training"):

    for name, model in models.items():

        print("\nTraining Model:", name)

        # Perform hyperparameter tuning
        grid = GridSearchCV(
            estimator=model,
            param_grid=parameters[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        # Train model using training data
        grid.fit(X_train, y_train)

        # Get best parameters and cross-validation accuracy
        model_params = grid.best_params_
        model_score = grid.best_score_

        print("Best Parameters:", model_params)
        print("Cross Validation Accuracy:", model_score)

        # Store tuned parameters for each model
        all_best_params[name] = model_params

        # Log model accuracy in MLflow
        mlflow.log_metric(name + "_cv_accuracy", model_score)

        # Log tuned parameters in MLflow
        for param, value in model_params.items():
            mlflow.log_param(name + "_" + param, value)

        # Save each trained model separately
        model_path = OUTPUT_DIR / (name + ".joblib")

        joblib.dump(grid.best_estimator_, model_path)

        print("Model Saved:", model_path)

        # Log each model file in MLflow
        mlflow.log_artifact(str(model_path))


    # 7. Save best hyperparameters of all three models
    params_path = OUTPUT_DIR / "best_params.json"

    with open(params_path, "w") as file:
        json.dump(all_best_params, file, indent=4)

    # Log parameters JSON file in MLflow
    mlflow.log_artifact(str(params_path))


# 8. Display final message
print("\nAll three models trained and saved successfully!")
print("Parameters Saved:", params_path)
