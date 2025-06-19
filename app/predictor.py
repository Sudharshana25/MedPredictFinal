import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("../models/diabetes_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Feature order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
input_data = []

print("🔍 Enter patient details for prediction:")
fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for field in fields:
    value = float(input(f"{field}: "))
    input_data.append(value)

# Convert & scale input
input_array = np.array([input_data])
input_scaled = scaler.transform(input_array)

# Predict
prediction = model.predict(input_scaled)[0]

# Output
if prediction == 1:
    print("\n⚠️ Likely to have diabetes (Positive Prediction)")
else:
    print("\n✅ Not likely to have diabetes (Negative Prediction)")
