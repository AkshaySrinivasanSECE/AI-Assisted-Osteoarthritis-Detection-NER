# Model and evaluation

## Prediction task

The supervised binary classifier estimates model probabilities for healthy (`group = 0`) and knee-OA (`group = 1`) patterns from age, gender, BMI, and VAS score. The prediction is a screening output, not a medical diagnosis.

The production feature order is `age`, `gender`, `BMI`, `VAS score`. VAS remains an integer-valued input to estimators and is categorical only in SMOTENC's sampling mask, alongside gender. WOMAC is not part of the production model.

## Candidate models

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

Exact configurations and the fixed random seed are stored in `models/model_metadata.json`.

## Cross-validation

Candidate comparison uses shuffled stratified 5-fold cross-validation over the unified active dataset. For each outer fold:

1. Active training rows are separated from untouched active validation rows.
2. Each candidate trains directly on the active training partition.
3. Observed and synthetic rows are treated identically by the estimator; source labels are retained only for audit counts.
4. Accuracy, precision, recall, and F1 are calculated on the untouched active validation rows.

Per-fold results are preserved in `docs/results/model_fold_metrics.csv`. `docs/results/validation_fold_assignments.csv` demonstrates that every active row is held out once and retains its provenance label.

## Selection and final fit

The model family with the highest mean outer-fold F1 is selected from the unified active-data comparison. The current artifact selects KNN; its comparison and fold metrics are authoritative in `docs/results/model_comparison.csv` and `docs/results/model_fold_metrics.csv`. The separate controlled feature experiment artifacts are historical research outputs over the preserved observed table and are not used to exclude the generated rows from production training.

The final saved model is fitted using the selected configuration recorded in metadata. Its serialized feature order is validated against the estimator's fitted input names and the API feature order.

The authoritative comparison is `docs/results/model_comparison.csv`; copied metric tables are intentionally omitted from this document so they cannot drift from retrained artifacts.

## Probability calibration

Sigmoid calibration is evaluated rather than assumed to improve probabilities. Calibration data contains active rows selected only from the current outer training partition. Adoption requires improvements in out-of-fold Brier score and log loss without reducing out-of-fold F1. The decision and comparison are stored in metadata and `docs/results/calibration_comparison.csv`.

## Explainability and importance

Two related but distinct outputs are available:

- The prediction endpoint explains the selected fitted model. Logistic Regression uses standardized coefficient contributions; other estimators use model-based feature ablation against active-dataset baselines.
- Training produces permutation feature importance on untouched active outer-fold validation records. These values are saved in `docs/results/feature_importance.csv` and its fold-level companion.

The model overview API also exposes normalized absolute standardized coefficients as global model importance. None of these explanations demonstrate causation.

## Confusion matrix

The selected model confusion matrix is based on all active out-of-fold predictions and is stored in `docs/results/confusion_matrix.csv`.

## Evidence boundary

- Evaluation is internal cross-validation on a small observed dataset augmented into a 588-row active table.
- There is no independent external or prospective clinical validation.
- Synthetic examples are generated rows in the unified model table, not patients or additional clinical evidence.
- Probability categories and recommendations are prototype presentation logic.
- Model explanations describe learned associations and should not be interpreted as medical causes.
