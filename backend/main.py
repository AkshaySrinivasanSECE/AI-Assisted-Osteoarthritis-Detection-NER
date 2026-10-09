from pathlib import Path
from typing import Literal
import hashlib
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.explainability import build_explainer


ROOT_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DATA_PATH = ROOT_DIR / "data" / "full_indicators_summary.csv"
MODEL_PATH = ROOT_DIR / "models" / "oa_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "features.pkl"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
MODEL_COMPARISON_PATH = ROOT_DIR / "docs" / "results" / "model_comparison.csv"
CONFUSION_MATRIX_PATH = ROOT_DIR / "docs" / "results" / "confusion_matrix.csv"

DEVELOPMENT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

logger = logging.getLogger("oa_scan_api")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PatientData(StrictModel):
    age: int = Field(ge=18, le=100)
    gender: Literal[0, 1]
    BMI: float = Field(gt=0, le=80)
    VAS_score: int = Field(ge=0, le=10)


class FeatureValues(StrictModel):
    age: int
    gender: Literal[0, 1]
    BMI: float
    VAS_score: int


class GlobalImportanceItem(StrictModel):
    feature: str
    standardized_coefficient: float
    direction_when_feature_increases: Literal["toward_oa", "toward_healthy", "neutral"]
    relative_importance: float


class LocalContributionItem(StrictModel):
    feature: str
    input_value: float
    contribution_direction: Literal["toward_oa", "toward_healthy", "neutral"]
    relative_contribution: float
    log_odds_contribution: float


class ExplanationResponse(StrictModel):
    explanation_type: str
    target_explained: str
    global_importance_method: str
    local_contribution_method: str
    direction_definition: dict[str, str]
    baseline_log_odds: float
    prediction_log_odds: float
    global_feature_importance: list[GlobalImportanceItem]
    local_feature_contributions: list[LocalContributionItem]
    disclaimer: str


class PredictionResponse(StrictModel):
    prediction: Literal[0, 1]
    result: str
    risk_level: Literal["Low", "Moderate", "High"]
    healthy_probability: float
    oa_probability: float
    model_name: str
    model_version: str
    feature_values_used: FeatureValues
    explanation: ExplanationResponse
    message: str
    disclaimer: str


class HealthResponse(StrictModel):
    api_status: Literal["healthy", "degraded"]
    model_loaded: bool


class ValidationStrategy(StrictModel):
    name: str
    folds: int
    stratified: bool
    shuffle: bool
    random_seed: int
    validation_subjects: str
    synthetic_validation_rows: int


class ModelInfoResponse(StrictModel):
    selected_model: str
    features: list[str]
    original_subject_count: int
    synthetic_training_count: int
    total_training_rows: int
    validation_strategy: ValidationStrategy
    model_version: str
    training_timestamp: str | None


class CandidateEvaluation(StrictModel):
    model: str
    accuracy: float
    accuracy_std: float | None
    precision: float
    precision_std: float | None
    recall: float
    recall_std: float | None
    f1: float
    f1_std: float | None


class ConfusionMatrixResponse(StrictModel):
    labels: list[str]
    matrix: list[list[int]]
    evaluated_on: str


class EvaluationResponse(StrictModel):
    candidate_models: list[CandidateEvaluation]
    selected_model: str
    validation_strategy: str
    confusion_matrix: ConfusionMatrixResponse
    global_feature_importance_method: str
    global_feature_importance: list[GlobalImportanceItem]
    disclaimer: str


app = FastAPI(
    title="OA Risk Prediction API",
    description="AI-assisted knee osteoarthritis risk-screening prototype",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEVELOPMENT_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


model = None
features = []
explainer = None
model_metadata = None
model_comparison = None
confusion_matrix_table = None
model_load_error = None
metadata_load_error = None
evaluation_load_error = None
model_name = "Unavailable"
model_version = "unavailable"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resources():
    global model, features, explainer, model_metadata
    global model_comparison, confusion_matrix_table
    global model_load_error, metadata_load_error, evaluation_load_error
    global model_name, model_version

    try:
        model = joblib.load(MODEL_PATH)
        features = list(joblib.load(FEATURES_PATH))
        active_data = pd.read_csv(ACTIVE_DATA_PATH, encoding="utf-8-sig")
        explainer = build_explainer(model, features, active_data)
        model_version = f"oa-model-{file_sha256(MODEL_PATH)[:12]}"
    except Exception as exc:  # Keep /health available when artifacts cannot load.
        model_load_error = str(exc)
        logger.exception("Model resources could not be loaded.")

    try:
        model_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        model_name = model_metadata["model_selection"]["selected_model"]
        model_version = model_metadata.get("model_version", model_version)
    except Exception as exc:
        metadata_load_error = str(exc)
        logger.exception("Model metadata could not be loaded.")

    try:
        model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
        confusion_matrix_table = pd.read_csv(CONFUSION_MATRIX_PATH)
        required_comparison_columns = {
            "Model",
            "Accuracy Mean",
            "Accuracy Std",
            "Precision Mean",
            "Precision Std",
            "Recall Mean",
            "Recall Std",
            "F1 Mean",
            "F1 Std",
        }
        missing = sorted(required_comparison_columns - set(model_comparison.columns))
        if missing:
            raise ValueError(f"Model comparison is missing columns: {missing}")
        if confusion_matrix_table.shape != (2, 3):
            raise ValueError("Expected a 2x2 confusion matrix table with an actual-label column.")
    except Exception as exc:
        evaluation_load_error = str(exc)
        logger.exception("Evaluation artifacts could not be loaded.")


load_resources()


def require_model():
    if model is None or explainer is None or not features:
        raise HTTPException(status_code=503, detail="The prediction model is not available.")


def require_metadata():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata is not available.")


def require_evaluation():
    if model_comparison is None or confusion_matrix_table is None:
        raise HTTPException(status_code=503, detail="Evaluation results are not available.")


def optional_float(value):
    return None if pd.isna(value) else float(value)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled API error for %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal API error occurred."},
    )


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "OA Risk Prediction API is running",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = model is not None and explainer is not None and bool(features)
    return {
        "api_status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    require_model()
    require_metadata()
    data = model_metadata["data"]
    validation = model_metadata["cross_validation"]
    return {
        "selected_model": model_name,
        "features": features,
        "original_subject_count": int(data["original_observed_subjects"]),
        "synthetic_training_count": int(data["synthetic_training_examples"]),
        "total_training_rows": int(data["active_training_rows"]),
        "validation_strategy": {
            "name": validation["method"],
            "folds": int(validation["n_splits"]),
            "stratified": validation["method"] == "StratifiedKFold",
            "shuffle": bool(validation["shuffle"]),
            "random_seed": int(validation["random_seed"]),
            "validation_subjects": validation["outer_validation_subjects"],
            "synthetic_validation_rows": int(validation["outer_validation_synthetic_rows"]),
        },
        "model_version": model_version,
        "training_timestamp": model_metadata.get("training_timestamp"),
    }


@app.get("/evaluation", response_model=EvaluationResponse)
def evaluation():
    require_model()
    require_metadata()
    require_evaluation()
    candidates = []
    for _, row in model_comparison.iterrows():
        candidates.append({
            "model": str(row["Model"]),
            "accuracy": float(row["Accuracy Mean"]),
            "accuracy_std": optional_float(row["Accuracy Std"]),
            "precision": float(row["Precision Mean"]),
            "precision_std": optional_float(row["Precision Std"]),
            "recall": float(row["Recall Mean"]),
            "recall_std": optional_float(row["Recall Std"]),
            "f1": float(row["F1 Mean"]),
            "f1_std": optional_float(row["F1 Std"]),
        })

    matrix = confusion_matrix_table[
        ["Predicted Healthy", "Predicted Knee OA"]
    ].astype(int).values.tolist()
    return {
        "candidate_models": candidates,
        "selected_model": model_name,
        "validation_strategy": (
            "Stratified 5-fold cross-validation on the unified 588-row active dataset; "
            "observed and synthetic rows are modeled together."
        ),
        "confusion_matrix": {
            "labels": ["Healthy", "Knee OA"],
            "matrix": matrix,
            "evaluated_on": "all active out-of-fold rows",
        },
        "global_feature_importance_method": (
            "Model-based feature importance from the selected fitted estimator."
        ),
        "global_feature_importance": explainer.global_feature_importance,
        "disclaimer": (
            "These are internal prototype cross-validation results, not external clinical "
            "validation. Synthetic rows are generated examples, not additional patients or "
            "independent clinical evidence."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    require_model()
    patient = pd.DataFrame([{
        "age": data.age,
        "gender": data.gender,
        "BMI": data.BMI,
        "VAS score": data.VAS_score,
    }])[features]

    try:
        prediction = int(model.predict(patient)[0])
        probabilities = model.predict_proba(patient)[0]
        model_classes = list(model.classes_)
        healthy_probability = round(float(probabilities[model_classes.index(0)]) * 100, 2)
        oa_probability = round(float(probabilities[model_classes.index(1)]) * 100, 2)
        explanation = explainer.explain(patient)
    except Exception as exc:
        logger.exception("Prediction failed.", exc_info=exc)
        raise HTTPException(
            status_code=500,
            detail="The prediction could not be completed.",
        ) from exc

    if oa_probability < 35:
        risk_level = "Low"
    elif oa_probability < 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "prediction": prediction,
        "result": "OA Risk Detected" if prediction == 1 else "Low OA Risk",
        "risk_level": risk_level,
        "healthy_probability": healthy_probability,
        "oa_probability": oa_probability,
        "model_name": model_name,
        "model_version": model_version,
        "feature_values_used": {
            "age": data.age,
            "gender": data.gender,
            "BMI": data.BMI,
            "VAS_score": data.VAS_score,
        },
        "explanation": explanation,
        "message": (
            "Further clinical assessment may be recommended."
            if prediction == 1
            else "No strong OA risk pattern was detected by the current model."
        ),
        "disclaimer": (
            "This is an AI-assisted screening prototype and not a medical diagnosis."
        ),
    }
