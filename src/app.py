# src/app.py
import streamlit as st
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

st.set_page_config(page_title="Synthetic Medical Records — Disease Predictor", layout="centered")

@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    return package["model"], package["label_encoder"], package["features"]

model, label_encoder, FEATURES = load_model()

st.title("🩺 Synthetic Medical Records — Disease Predictor")
st.markdown(
    "Enter patient vitals below to get a predicted diagnosis. "
    "Model trained on synthetic, privacy-safe records."
)

with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60.0, max_value=220.0, value=120.0, format="%.1f")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80.0, max_value=400.0, value=180.0, format="%.1f")
    glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=100.0, format="%.1f")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, format="%.1f")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=180.0, value=72.0, format="%.1f")

    submitted = st.form_submit_button("Predict")

if submitted:
    X = np.array([[age, blood_pressure, cholesterol, glucose, bmi, heart_rate]])
    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(X)[0]
    proba_dict = dict(zip(label_encoder.classes_, [float(round(p, 4)) for p in proba]))

    st.subheader(f"Predicted diagnosis: **{pred_label}**")
    st.write("Probabilities:")
    st.json(proba_dict)

    st.write("---")
    st.write("Model info:")
    st.write(f"- Features used: {FEATURES}")
    st.write(f"- Model type: {type(model).__name__}")
