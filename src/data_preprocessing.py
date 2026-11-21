"""Stage 4: Fit preprocessing on training features only."""
from pathlib import Path
import pandas as pd
import mlflow
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "artifacts/split"
OUT = ROOT / "artifacts/processed"
OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(SPLIT / "train.csv")
test = pd.read_csv(SPLIT / "test.csv")

# Keep labels separate; never encode Loan_ID as a feature.
X_train = train.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
X_test = test.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
numeric = X_train.select_dtypes(include="number").columns.tolist()
categorical = X_train.select_dtypes(exclude="number").columns.tolist()

# Numerical: fill missing with median, then standardize.
num_steps = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# Categorical: fill missing with most frequent value, then one-hot encode.
cat_steps = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
transformer = ColumnTransformer([
    ("numeric", num_steps, numeric),
    ("categorical", cat_steps, categorical)
], sparse_threshold=0)

# Learn medians, categories, and scaling from train ONLY.
train_array = transformer.fit_transform(X_train)
test_array = transformer.transform(X_test)
feature_names = transformer.get_feature_names_out()
pd.DataFrame(train_array, columns=feature_names).to_csv(OUT / "train_features.csv", index=False)
pd.DataFrame(test_array, columns=feature_names).to_csv(OUT / "test_features.csv", index=False)
joblib.dump(transformer, OUT / "preprocessor.joblib")
mlflow.set_experiment("Loan_Data_Pipeline")
with mlflow.start_run(run_name="Data_Preprocessing"):
    mlflow.log_metric("processed_features", len(feature_names))
    mlflow.log_metric("train_rows", len(train_array))
    mlflow.log_metric("test_rows", len(test_array))
    mlflow.log_artifact(str(OUT / "preprocessor.joblib"))
print("Processed train and test features saved:", len(feature_names), "columns")
