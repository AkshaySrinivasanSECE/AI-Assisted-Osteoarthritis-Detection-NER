from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "oa_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "features.pkl"


# --------------------------------------------------
# CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="OA Risk Prediction API",
    description="AI-Assisted Early Detection System for Osteoarthritis Risk Markers",
    version="1.0"
)


# --------------------------------------------------
# ENABLE CORS FOR REACT FRONTEND
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


# --------------------------------------------------
# REQUEST DATA FORMAT
# --------------------------------------------------

class PatientData(BaseModel):
    age: float
    gender: int
    BMI: float
    VAS_score: float
    JPR_30: float
    JPR_45: float
    JPR_60: float


# --------------------------------------------------
# HOME ROUTE
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "OA Risk Prediction API is running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# PREDICTION ROUTE
# --------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    # Convert frontend input into dataframe
    patient = pd.DataFrame([
        {
            "age": data.age,
            "gender": data.gender,
            "BMI": data.BMI,
            "VAS score": data.VAS_score,
            "JPR_30": data.JPR_30,
            "JPR_45": data.JPR_45,
            "JPR_60": data.JPR_60
        }
    ])

    # Keep feature order exactly same as training
    patient = patient[features]

    # Make prediction
    prediction = int(model.predict(patient)[0])

    # Get probability
    probabilities = model.predict_proba(patient)[0]

    healthy_probability = round(
        float(probabilities[0]) * 100,
        2
    )

    oa_probability = round(
        float(probabilities[1]) * 100,
        2
    )

    # Basic screening classification
    if oa_probability < 35:
        risk_level = "Low"

    elif oa_probability < 70:
        risk_level = "Moderate"

    else:
        risk_level = "High"

    # Final response
    return {
        "prediction": prediction,

        "result": (
            "OA Risk Detected"
            if prediction == 1
            else "Low OA Risk"
        ),

        "risk_level": risk_level,

        "healthy_probability": healthy_probability,

        "oa_probability": oa_probability,

        "message": (
            "Further clinical assessment may be recommended."
            if prediction == 1
            else
            "No strong OA risk pattern was detected by the current model."
        ),

        "disclaimer": (
            "This is an AI-assisted screening prototype "
            "and not a medical diagnosis."
        )
    }