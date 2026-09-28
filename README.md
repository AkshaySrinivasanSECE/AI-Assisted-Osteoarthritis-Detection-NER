# OA-SCAN

AI-Assisted Early Detection System for Osteoarthritis Risk Markers in NER

## Project overview

OA-SCAN is a prototype AI-assisted screening dashboard designed to support early identification of knee osteoarthritis risk markers. The system uses a compact clinical dataset and a Random Forest classifier to estimate risk based on accessible clinical inputs such as age, gender, BMI, pain score, and joint position reproduction measures.

## Problem statement

Osteoarthritis risk assessment often depends on late-stage clinical evaluation. This project explores a lightweight risk-screening workflow that can support early identification in resource-constrained and underserved settings, with a focus on healthcare access in India’s North Eastern Region.

## Motivation

The goal is to create a research-oriented screening prototype that is easy to understand, easy to run, and suitable for hackathon presentation. The solution emphasizes accessible screening inputs and a clean, professional user interface rather than claiming a clinical-grade diagnostic system.

## Current prototype capabilities

- Patient screening form with structured inputs
- Real-time AI-assisted risk assessment through FastAPI
- Probability-based OA risk output
- Model comparison workflow for research evaluation
- Documentation for architecture, dataset, and model design
- Frontend presentation designed for demo use

## System architecture

React frontend → FastAPI REST API → preprocessing and feature ordering → Random Forest model → risk probability and result → frontend result card

## Dataset

- 88 subjects total
- 45 healthy
- 43 knee OA
- Target variable: `group`
- `group = 0`: healthy
- `group = 1`: knee osteoarthritis

## ML features

The current model uses the following predictors:

- `age`
- `gender`
- `BMI`
- `VAS score`
- `JPR_30`
- `JPR_45`
- `JPR_60`

These are the same features saved in `models/features.pkl`.

## Algorithms evaluated

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM

## Model comparison table

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.717 | 0.743 | 0.678 | 0.690 |
| Logistic Regression | 0.671 | 0.679 | 0.647 | 0.653 |
| SVM | 0.671 | 0.697 | 0.606 | 0.636 |
| KNN | 0.591 | 0.618 | 0.539 | 0.563 |
| Decision Tree | 0.578 | 0.595 | 0.533 | 0.529 |

## Why Random Forest was selected

Random Forest produced the strongest F1 score in the current cross-validation experiment and is therefore the selected model for the current prototype.

## Current evaluation metrics

The prototype metrics above should be treated as research results for a small dataset and not as clinical performance claims. They are useful for evaluating model behavior and comparing algorithms, but they do not establish medical-grade accuracy.

## Screenshots

Screenshots for demonstration can be added here when collected from the running app:

- Landing page
- Screening flow
- Result dashboard
- Model evaluation charts

## Backend API

The backend is implemented with FastAPI and exposes:

- `GET /`
- `GET /health`
- `POST /predict`

## Installation

From the project root:

```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to run backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload
```

Expected local URL:

- http://127.0.0.1:8000

## How to run frontend

```bash
cd frontend
npm install
npm run dev
```

Expected local URL:

- http://localhost:5173
- If port 5173 is occupied, Vite may use the next available port such as 5174.

## Example API request

```json
{
  "age": 60,
  "gender": 0,
  "BMI": 26.71,
  "VAS_score": 5,
  "JPR_30": 3.5,
  "JPR_45": 4.2,
  "JPR_60": 4.8
}
```

## Example API response

```json
{
  "prediction": 1,
  "result": "OA Risk Detected",
  "risk_level": "High",
  "healthy_probability": 24.75,
  "oa_probability": 75.25,
  "message": "Further clinical assessment may be recommended.",
  "disclaimer": "This is an AI-assisted screening prototype and not a medical diagnosis."
}
```

## Folder structure

```text
Aiml project/
├── backend/
│   └── main.py
├── data/
│   └── full_indicators_summary.csv
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── model.md
│   └── results/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
├── models/
│   ├── features.pkl
│   └── oa_model.pkl
├── .gitignore
├── README.md
├── requirements.txt
└── ...
```

## Limitations

- Small research dataset
- Prototype only
- No clinical validation
- Risk screening should not be treated as diagnosis
- Performance is not yet generalizable to larger populations

## Future scope

The following are future research and product ideas, not completed functionality:

- multilingual support
- offline-first screening
- larger clinically validated datasets
- sensor integration
- gait analysis
- explainable AI
- healthcare-worker dashboard
- NER-specific validation

## Medical disclaimer

This system is a prototype for AI-assisted screening and risk-marker detection. It is not intended to provide a medical diagnosis. Clinical evaluation by a qualified healthcare professional remains necessary.
