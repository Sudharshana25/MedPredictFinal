import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="MedPredict - Diabetes Risk", layout="centered")

st.title("🧠 MedPredict - Diabetes Risk Predictor")
st.markdown("Predict diabetes risk from patient test data.")

# Option: Single Input or CSV
option = st.radio("Choose input method:", ["📝 Manual Entry", "📁 CSV Upload"])

# ------------ SINGLE INPUT FORM ------------
if option == "📝 Manual Entry":
    st.subheader("Enter Patient Details:")

    fields = {
        "Pregnancies": st.number_input("Pregnancies", 0, 20, 1),
        "Glucose": st.number_input("Glucose", 0, 200, 120),
        "BloodPressure": st.number_input("BloodPressure", 0, 140, 70),
        "SkinThickness": st.number_input("SkinThickness", 0, 100, 20),
        "Insulin": st.number_input("Insulin", 0, 900, 85),
        "BMI": st.number_input("BMI", 0.0, 70.0, 25.0),
        "DiabetesPedigreeFunction": st.number_input("DiabetesPedigreeFunction", 0.0, 2.5, 0.5),
        "Age": st.number_input("Age", 0, 120, 35)
    }

    if st.button("🔍 Predict"):
        input_data = np.array([list(fields.values())])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        label = "⚠️ Diabetic" if prediction == 1 else "✅ Non-Diabetic"
        st.success(f"**Prediction:** {label}")

# ------------ BULK CSV UPLOAD ------------
else:
    st.subheader("Upload CSV File:")
    uploaded_file = st.file_uploader("Upload a CSV file without the 'Outcome' column", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        try:
            scaled_data = scaler.transform(df)
            predictions = model.predict(scaled_data)

            df['Prediction'] = predictions
            df['Prediction_Label'] = df['Prediction'].apply(lambda x: "Diabetic" if x == 1 else "Non-Diabetic")

            st.success("✅ Predictions completed:")
            st.dataframe(df)

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", csv_download, "diabetes_predictions.csv", "text/csv")

        except Exception as e:
            st.error("❌ Error processing file: make sure the format matches the training dataset.")
