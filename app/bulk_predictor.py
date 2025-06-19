import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("../models/diabetes_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Load dataset and drop the label column
df = pd.read_csv("../data/diabetes.csv")
if 'Outcome' in df.columns:
    df = df.drop(columns=['Outcome'])

# Scale input features
scaled_data = scaler.transform(df)

# Predict
predictions = model.predict(scaled_data)

# Add prediction to the DataFrame
df['Prediction'] = predictions
df['Prediction_Label'] = df['Prediction'].apply(lambda x: "Diabetic" if x == 1 else "Non-Diabetic")

# Save results
df.to_csv("../data/diabetes_predictions.csv", index=False)

print("✅ Predictions saved to diabetes_predictions.csv")
