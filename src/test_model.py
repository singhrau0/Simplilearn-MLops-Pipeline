
from pathlib import Path
import pandas as pd
import joblib


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "processed" / "preprocessor.joblib"
MODEL_PATH = BASE_DIR / "artifacts" / "champion" / "champion.joblib"


# 2. Load preprocessing pipeline
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Preprocessing pipeline loaded successfully!")


# 3. Load champion model
model = joblib.load(MODEL_PATH)

print("Champion model loaded successfully!")


# 4. Create sample user input
user_data = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}


# 5. Convert user input into DataFrame
input_df = pd.DataFrame([user_data])

print("\nOriginal User Data:")
print(input_df)


# 6. Preprocess user input
processed_data = preprocessor.transform(input_df)

# 7. Get original feature names from preprocessing
feature_names = preprocessor.get_feature_names_out()

# 8. Convert NumPy array into DataFrame
processed_df = pd.DataFrame(
    processed_data,
    columns=feature_names
)

print("Preprocessed Data Shape:", processed_df.shape)

# 9. Predict using champion model
prediction = model.predict(processed_df)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Loan Status: APPROVED")
elif prediction[0] == 0:
    print("Loan Status: REJECTED")
else:
    print("Loan Status:", prediction[0])


print("\nModel prediction completed successfully!")
