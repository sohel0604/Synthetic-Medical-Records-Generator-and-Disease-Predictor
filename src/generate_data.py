# src/generate_data.py
import random
from faker import Faker
import pandas as pd
import numpy as np
from pathlib import Path

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "synthetic_medical_records.csv"

N = 10000

def make_record():
    age = int(np.clip(np.random.normal(50, 18), 1, 100))  # realistic age spread
    gender = random.choice(["Male", "Female"])
    # Baseline vitals with noise
    systolic_bp = round(np.clip(np.random.normal(120 + (age-50)*0.4, 12), 80, 200), 1)
    cholesterol = round(np.clip(np.random.normal(180 + (age-50)*0.6, 25), 100, 350), 1)
    glucose = round(np.clip(np.random.normal(100 + (age-50)*0.5, 18), 60, 300), 1)
    heart_rate = round(np.clip(np.random.normal(72, 10), 40, 150), 1)
    bmi = round(np.clip(np.random.normal(26, 4.5), 12, 50), 1)
    # binary flags with some dependence
    has_diabetes_flag = 1 if glucose > 126 or (age > 55 and random.random() < 0.12) else 0
    smoker_flag = 1 if random.random() < 0.18 else 0

    # Diagnosis logic (probabilistic, not deterministic):
    # - If very high BP -> Hypertension
    # - If glucose suggests diabetes -> Diabetes
    # - If high cholesterol + age or smoking -> Heart Disease risk
    # - Else Healthy (with some noise)
    diag_scores = {
        "Hypertension": 0.0,
        "Diabetes": 0.0,
        "Heart Disease": 0.0,
        "Healthy": 0.0
    }

    # Compute heuristic scores
    diag_scores["Hypertension"] += max(0, (systolic_bp - 130) / 40)
    diag_scores["Diabetes"] += max(0, (glucose - 126) / 50) + 0.5*has_diabetes_flag
    diag_scores["Heart Disease"] += max(0, (cholesterol - 200) / 80) + 0.5*(smoker_flag) + 0.02*(age-50)
    diag_scores["Healthy"] += 1.0 - (diag_scores["Hypertension"] + diag_scores["Diabetes"] + diag_scores["Heart Disease"])
    # Add small randomness
    for k in diag_scores:
        diag_scores[k] += np.random.normal(0, 0.15)

    # Choose diagnosis by max score, but allow some chance to pick other label to introduce noise
    diagnosis = max(diag_scores, key=diag_scores.get)
    if random.random() < 0.06:  # 6% noise
        diagnosis = random.choice(["Healthy", "Hypertension", "Diabetes", "Heart Disease"])

    return {
        "Patient_ID": fake.uuid4(),
        "Name": fake.name(),
        "Age": age,
        "Gender": gender,
        "Blood_Pressure": systolic_bp,
        "Cholesterol": cholesterol,
        "Glucose": glucose,
        "Heart_Rate": heart_rate,
        "BMI": bmi,
        "Has_Diabetes": int(has_diabetes_flag),
        "Smoker": int(smoker_flag),
        "Diagnosis": diagnosis
    }

def generate(n=N, out_csv=OUT_CSV):
    records = [make_record() for _ in range(n)]
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} records to {out_csv}")

if __name__ == "__main__":
    generate()
