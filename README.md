# MedPredictFinal
# MedPredict - Disease Risk Predictor (Diabetes)

> A real-time machine learning system to predict diabetes risk from medical data — built using Python, scikit-learn, and Streamlit.

![medpredict-banner](https://imgur.com/aREbmQt.png)

---

## Features

- Predict **diabetes risk** using test values (Glucose, BMI, etc.)  
- Supports **manual input** or **bulk CSV upload**  
- Trained using **Random Forest, Logistic Regression, SVM, KNN**  
- Live predictions via **Streamlit Web App**  
- Downloadable prediction results  
- Model trained on trusted **Kaggle Diabetes Dataset**

---

## Machine Learning Models Used

- Logistic Regression  
- Random Forest 🌲 *(Best performer)*  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

Evaluated using:
- Accuracy, Precision, Recall
- Confusion Matrix
- Classification Report

---

##  Sample Input (Manual or CSV)

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age |
|-------------|---------|----------------|---------------|---------|------|---------------------------|-----|
| 6           | 148     | 72             | 35            | 0       | 33.6 | 0.627                     | 50  |

---

## Web App (Live Demo)

👉 https://medpredictfinal-phuak76z9krrn4zxqtdvxc.streamlit.app/

---


## Tech Stack

- Python 3.10
- scikit-learn
- pandas
- NumPy
- Streamlit
- joblib

---

## Installation & Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/MedPredict.git
cd MedPredict

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Author 
Sudharshana J BE CSE
GitHub: @Sudharshana25
