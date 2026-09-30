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

Candidate comparison uses shuffled stratified 5-fold cross-validation over the original subjects only. For each outer fold:

1. The original training subjects are separated from untouched original validation subjects.
2. SMOTENC fits only the training partition and adds the configured examples per class.
3. Each candidate trains on that fold's augmented training table.
4. Accuracy, precision, recall, and F1 are calculated only on the untouched original validation subjects.

Per-fold results are preserved in `docs/results/model_fold_metrics.csv`. `docs/results/validation_fold_assignments.csv` demonstrates that every validation row is real and each original subject is held out once.

## Selection and final fit

The model family with the highest mean outer-fold F1 is selected. The controlled feature experiments used the same stratified folds, models, and fold-local augmentation. The four-feature baseline with categorical VAS selected Logistic Regression (`mean F1 0.727333`, sample SD `0.048763`); numeric VAS selected Logistic Regression (`0.700000`, SD `0.057735`), while the best WOMAC candidate was Random Forest (`0.713571`, SD `0.070448`). The production decision therefore retains the four-feature categorical-VAS baseline. Full experiment methods, fold metrics, and subject-level predictions are in `docs/results/controlled_experiments/`.

The final saved model is fitted using the selected configuration recorded in metadata. Its serialized feature order is validated against the estimator's fitted input names and the API feature order.

The authoritative comparison is `docs/results/model_comparison.csv`; copied metric tables are intentionally omitted from this document so they cannot drift from retrained artifacts.

## Probability calibration

Sigmoid calibration is evaluated rather than assumed to improve probabilities. Calibration data contains original subjects selected only from the current outer training partition. Adoption requires improvements in out-of-fold Brier score and log loss without reducing out-of-fold F1. The decision and comparison are stored in metadata and `docs/results/calibration_comparison.csv`.

## Explainability and importance

Two related but distinct outputs are available:

- The prediction endpoint explains a Logistic Regression prediction using standardized feature values multiplied by trained coefficients. Relative absolute contributions describe the model calculation for that input.
- Training produces permutation feature importance on untouched original outer-fold validation records. These values are saved in `docs/results/feature_importance.csv` and its fold-level companion.

The model overview API also exposes normalized absolute standardized coefficients as global model importance. None of these explanations demonstrate causation.

## Confusion matrix

The selected model confusion matrix is based on real out-of-fold predictions and is stored in `docs/results/confusion_matrix.csv`. Synthetic training rows are not counted in the matrix.

## Evidence boundary

- Evaluation is internal cross-validation on a small original dataset.
- There is no independent external or prospective clinical validation.
- Synthetic examples are augmentation, not patients or additional evidence.
- Probability categories and recommendations are prototype presentation logic.
- Model explanations describe learned associations and should not be interpreted as medical causes.
