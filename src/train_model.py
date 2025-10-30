# src/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "synthetic_medical_records.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"

FEATURES = ["Age", "Blood_Pressure", "Cholesterol", "Glucose", "BMI", "Heart_Rate"]

def load_data():
    df = pd.read_csv(DATA_CSV)
    # Drop rows with NaNs if any (shouldn't be)
    df = df.dropna(subset=FEATURES + ["Diagnosis"])
    return df

def train_save(random_state=42):
    df = load_data()
    X = df[FEATURES].values
    y = df["Diagnosis"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save both classifier and label encoder
    joblib.dump({"model": clf, "label_encoder": le, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model package to {MODEL_PATH}")

    return clf, le

if __name__ == "__main__":
    train_save()
