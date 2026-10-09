# OA-SCAN

AI-assisted knee osteoarthritis risk-screening prototype for a college-level AIML demonstration.

## Project objective

OA-SCAN demonstrates an explainable, reproducible tabular machine-learning workflow for screening patterns associated with knee osteoarthritis. A user enters age, gender, height, weight, and pain score; the frontend calculates BMI and sends only the four trained model features to FastAPI.

The result is a model-generated screening probability and explanation. It is not a diagnosis, clinical recommendation system, or replacement for assessment by a qualified healthcare professional.

## Architecture

```text
React + Vite frontend
        |
        | GET /model-info, GET /evaluation, POST /predict
        v
FastAPI + Pydantic backend
        |
        | fixed feature order: age, gender, BMI, VAS score
        v
Saved scikit-learn model and metadata
        |
        +--> probabilities and risk category
        +--> model-based per-prediction explanation
        +--> evaluation and provenance artifacts
```

The data and training path is separate from inference:

```text
Preserved original CSV
        -> validated, provenance-tagged SMOTENC generation
        -> active training CSV and quality report
        -> unified active-dataset outer cross-validation
        -> candidate-model comparison and selection
        -> saved model, metadata, tables, and plots
```

More detail is available in [docs/architecture.md](docs/architecture.md).

## Technologies

- Python
- Pandas and NumPy
- scikit-learn
- imbalanced-learn and SMOTENC
- Joblib
- FastAPI and Pydantic
- React and Vite
- CSS-based responsive visualizations
- Python `unittest` and Node's built-in test runner

No neural network or external explainability framework is used.

## Dataset

The project maintains three distinct counts:

- **88 original observed subjects**: 45 healthy and 43 knee OA labels.
- **500 synthetic training examples**: 250 per target class.
- **588 active training rows**: the original and synthetic rows combined for final training.

These counts come from the active data artifacts and are recorded in [models/model_metadata.json](models/model_metadata.json) and [data/synthetic_data_quality_report.json](data/synthetic_data_quality_report.json).

The target is `group`:

- `0`: healthy
- `1`: knee osteoarthritis

The production model uses the four selected features, in this order:

- `age`
- `gender`
- `BMI`
- `VAS score`

The production comparison uses the unified active dataset and currently selects KNN by mean F1. VAS remains an integer-valued estimator input. The separate controlled feature experiments in `docs/results/controlled_experiments/` are historical analyses of the preserved observed table and do not control production inclusion of generated rows.

The preserved source is `data/full_indicators_summary.original.csv`. The generated model-ready table is `data/full_indicators_summary.csv`, and all 588 rows are passed to the model as one unified active dataset. Every active row includes `record_id`, `record_source`, `generation_method`, and `validation_eligible` provenance fields; these fields are used for audit and reporting, not to exclude generated rows from modeling.

Synthetic rows are generated examples, not additional patients, observed subjects, or independent clinical evidence.

## Synthetic data methodology

`ml/generate_synthetic_data.py` performs the following reproducible process:

1. Reads the preserved 88-subject source rather than reusing an earlier generated table.
2. Validates required fields, missing values, feature ranges, target labels, and categorical values.
3. Applies SMOTENC with a fixed random seed. Gender and VAS score are treated as categorical inputs.
4. Removes candidates that duplicate an original feature vector or another synthetic feature vector.
5. Selects exactly 250 unique examples for each target class.
6. Labels original rows as `real` and generated rows as `synthetic` for provenance.
7. Marks all active rows as eligible for the unified model's validation folds.
8. Replaces the generated active dataset on every run, so synthetic examples do not accumulate.
9. Writes a machine-readable quality and distribution report.

The generator does not modify the preserved original CSV.

## ML algorithms

The training pipeline compares:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

The current selected family is read from `models/model_metadata.json`; the frontend does not embed the selected-model name or evaluation scores.

## Validation methodology

Evaluation uses stratified 5-fold cross-validation with shuffling and the recorded fixed random seed.

- The outer folds contain all 588 active rows: 88 observed and 500 generated.
- Each active row is held out exactly once.
- No fold-local SMOTENC augmentation is applied during model evaluation; the generated rows are already part of the active dataset.
- Synthetic rows can appear in training, validation, calibration, and importance folds as ordinary active rows.
- Accuracy, precision, recall, and F1 are stored for every fold.
- The comparison reports the mean and sample standard deviation across folds.

This is internal cross-validation, not external or clinical validation. Fold-level evidence is stored in `docs/results/model_fold_metrics.csv` and `docs/results/validation_fold_assignments.csv`.

## Model selection

The candidate with the highest mean cross-validation F1 is selected. The current artifact identifies KNN as the selected family on the unified active dataset.

A sigmoid calibration candidate is evaluated on active rows drawn only from each outer training partition. Calibration is adopted only when Brier score and log loss improve without reducing out-of-fold F1. The current metadata records whether calibration was applied to the saved model.

Authoritative outputs are generated by training rather than copied into this README:

- [Model comparison](docs/results/model_comparison.csv)
- [Per-fold metrics](docs/results/model_fold_metrics.csv)
- [Confusion matrix](docs/results/confusion_matrix.csv)
- [Held-out permutation importance](docs/results/feature_importance.csv)
- [Calibration comparison](docs/results/calibration_comparison.csv)
- [Model metadata](models/model_metadata.json)

## API

The FastAPI service exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API and model-loaded status |
| `GET` | `/model-info` | Selected model, feature order, provenance counts, version, and validation configuration |
| `GET` | `/evaluation` | Candidate metrics, confusion matrix, selected model, and global model importance |
| `POST` | `/predict` | Prediction, probabilities, risk category, model metadata, exact inputs, and explanation |

Example prediction request:

```json
{
  "age": 60,
  "gender": 0,
  "BMI": 25.0,
  "VAS_score": 5
}
```

The response values must be read from the running API. This documentation intentionally does not contain a copied prediction or probability that could become stale after retraining.

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

## Frontend

The React interface provides:

- Step-based screening form validation.
- Automatic BMI calculation from height and weight.
- Prediction loading and API error states.
- OA and healthy probabilities from `/predict`.
- Risk category and conservative next-step wording.
- Model-based contribution direction and relative contribution for all four inputs.
- Model metadata and provenance from `/model-info`.
- Candidate comparison, validation method, confusion matrix, and importance from `/evaluation`.
- Responsive and keyboard-accessible layouts.

Supporting symptom and history answers are collected for the form workflow but are not sent to the classifier.

## Installation

From the project root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
cd ..
```

If PowerShell activation is restricted, commands can call `.\.venv\Scripts\python.exe` directly.

## Execution commands

Start the backend in terminal 1 from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in terminal 2:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

To regenerate data and train from the preserved source:

```powershell
.\.venv\Scripts\python.exe ml/generate_synthetic_data.py
.\.venv\Scripts\python.exe ml/train_model.py
```

`ml/train_model.py` creates the evaluation tables and plots as part of the same reproducible training command. `ml/evaluate_model.py` can be used afterward to regenerate plots from saved result tables.

Run the complete verification suite:

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

## Demonstration workflow

Use this sequence for a clean project demonstration:

1. Start the backend and show that `GET /health` reports the API healthy and the model loaded.
2. Start the frontend and open `http://127.0.0.1:5173`.
3. Select **Start Screening**.
4. Enter sample basic details: age `60`, female, height `165 cm`, and weight `68 kg`. Confirm that BMI is calculated automatically.
5. Continue to symptoms, keep the VAS pain score at `5`, and choose valid supporting answers.
6. Complete the medical-history step and submit the screening.
7. During loading, explain that only age, gender, BMI, and VAS score are sent to `/predict`.
8. On the result dashboard, show the OA probability, healthy probability, and risk category returned by the saved model.
9. Show the four per-prediction contribution cards and explain that they describe model behavior, not causation.
10. Show the selected model, model version, validation summary, original-subject count, and synthetic-training count.
11. Use the **AI Model** navigation link or scroll to the AI model section.
12. Show the model comparison table. Its values come from `GET /evaluation`, which reads the generated comparison artifact.
13. Show the unified stratified validation methodology and the generated-row count included in validation.
14. Show the dataset provenance cards, confusion matrix, and global feature-importance visualization.
15. Finish with the evidence boundary and limitations: small original sample, internal cross-validation only, generated rows are not clinical evidence, and the output is not a diagnosis.

Do not quote a memorized probability during the demonstration. The displayed prediction and evaluation values should always come from the currently loaded artifacts.

## Limitations

- Only 88 original subjects are available for validation.
- The 500 generated rows are included in the unified model table but cannot substitute for observed patients.
- There is no external test cohort or prospective clinical validation.
- Cross-validation results may be optimistic or unstable for a small dataset.
- Dataset representation, particularly categorical distributions, may not generalize to other populations.
- The four-feature model omits imaging, physical examination, longitudinal history, and other clinically relevant information.
- Risk thresholds are prototype presentation categories, not clinically validated decision thresholds.
- Feature contributions describe the fitted model and do not demonstrate causal risk factors.
- Supporting form answers beyond the four trained features do not affect prediction.

## Ethical and medical disclaimer

OA-SCAN is an educational AI-assisted screening prototype. It must not be used to diagnose knee osteoarthritis, make treatment decisions, replace professional judgment, or delay appropriate medical care.

Predictions, probabilities, feature contributions, and internal validation metrics describe this model and dataset only. They are not proof of clinical effectiveness. Anyone with persistent or concerning symptoms should seek assessment from a qualified healthcare professional.
