"""
Vercel Python serverless function (FastAPI ASGI app).
Vercel auto-detects the `app` variable in api/index.py and routes
all /api/* requests to it.
"""

import json
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

# Loaded once per cold start, reused across warm invocations.
_model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    _feature_columns = json.load(f)

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: str = Field(..., description="'M' or 'F'")
    chest_pain_type: str = Field(..., description="ATA, NAP, ASY, or TA")
    resting_bp: int = Field(..., ge=0, le=300, description="Resting blood pressure, mm Hg")
    cholesterol: int = Field(..., ge=0, le=700, description="Serum cholesterol, mm/dl")
    fasting_bs: int = Field(..., ge=0, le=1, description="1 if fasting blood sugar > 120 mg/dl, else 0")
    resting_ecg: str = Field(..., description="Normal, ST, or LVH")
    max_hr: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exercise_angina: str = Field(..., description="'Y' or 'N'")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression induced by exercise")
    st_slope: str = Field(..., description="Up, Flat, or Down")


def build_feature_vector(payload: PatientInput) -> np.ndarray:
    row = {col: 0 for col in _feature_columns}

    sex = payload.sex.strip().upper()
    if sex not in ("M", "F"):
        raise HTTPException(400, "sex must be 'M' or 'F'")
    row["Sex"] = 1 if sex == "M" else 0

    angina = payload.exercise_angina.strip().upper()
    if angina not in ("Y", "N"):
        raise HTTPException(400, "exercise_angina must be 'Y' or 'N'")
    row["ExerciseAngina"] = 1 if angina == "Y" else 0

    row["Age"] = payload.age
    row["RestingBP"] = payload.resting_bp
    row["Cholesterol"] = payload.cholesterol
    row["FastingBS"] = payload.fasting_bs
    row["MaxHR"] = payload.max_hr
    row["Oldpeak"] = payload.oldpeak

    cpt = payload.chest_pain_type.strip().upper()
    if cpt not in ("ATA", "NAP", "ASY", "TA"):
        raise HTTPException(400, "chest_pain_type must be one of ATA, NAP, ASY, TA")
    if cpt != "ASY":  # ASY was the dropped reference category during training
        key = f"ChestPainType_{cpt}"
        if key in row:
            row[key] = 1

    ecg = payload.resting_ecg.strip().capitalize()
    ecg_map = {"Normal": "Normal", "St": "ST", "Lvh": "LVH"}
    ecg = ecg_map.get(ecg, payload.resting_ecg.strip().upper())
    if ecg not in ("Normal", "ST", "LVH"):
        raise HTTPException(400, "resting_ecg must be one of Normal, ST, LVH")
    if ecg != "LVH":  # LVH was the dropped reference category during training
        key = f"RestingECG_{ecg}"
        if key in row:
            row[key] = 1

    slope = payload.st_slope.strip().capitalize()
    if slope not in ("Up", "Flat", "Down"):
        raise HTTPException(400, "st_slope must be one of Up, Flat, Down")
    if slope != "Down":  # Down was the dropped reference category during training
        key = f"ST_Slope_{slope}"
        if key in row:
            row[key] = 1

    vector = np.array([[row[col] for col in _feature_columns]], dtype=float)
    return vector


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/api/predict")
def predict(payload: PatientInput):
    vector = build_feature_vector(payload)
    scaled = _scaler.transform(vector)

    prediction = int(_model.predict(scaled)[0])
    probability = float(_model.predict_proba(scaled)[0][1])

    return {
        "prediction": prediction,
        "label": "Heart Disease Likely" if prediction == 1 else "No Heart Disease Indicated",
        "probability": round(probability, 4),
    }
