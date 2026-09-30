# Controlled OA feature experiments

## Scope and data

- Preserved observed dataset: `data/full_indicators_summary.original.csv` (SHA-256 `75bd6e52bf3950c8319efa20a72affbfe381685944f092530e10644e6813ca27`).
- Observed subjects: 88 (45 healthy, 43 OA).
- Target: `group` (`0` healthy, `1` OA).
- No production model, feature artifact, API, frontend, or active synthetic dataset was changed.
- No K-L grade, KOA period, JPR, EMG, or other excluded feature was used.

## Exact methodology

All configurations use the same shuffled `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` assignments and the same five production candidate models with unchanged hyperparameters. In every fold, SMOTENC is fitted only to that fold's original observed training partition and adds 250 examples per target class (500 total). Validation contains only untouched original subjects and zero synthetic rows. Metrics are computed separately per fold; reported standard deviations are sample standard deviations across the five folds (`ddof=1`). Selection is strictly the highest mean fold F1.

In this codebase, categorical representation refers to SMOTENC's categorical feature mask. The estimator pipelines do not one-hot encode gender or VAS; they receive the postprocessed numeric values exactly as in the current pipeline. VAS is rounded back to its integer score after augmentation in both VAS variants, preserving existing postprocessing. WOMAC is numeric in SMOTENC and rounded to its observed integer-score representation after augmentation.

## Configurations

- `vas_categorical_baseline`: age, gender, BMI, VAS score; SMOTENC categorical fields: gender and VAS score.
- `vas_numeric_baseline`: age, gender, BMI, VAS score; SMOTENC categorical field: gender only.
- `womac_candidate`: age, gender, BMI, VAS score, WOMAC score; SMOTENC categorical fields: gender and VAS score.

## Summary metrics

| Configuration | Model | Accuracy Mean | Accuracy Std | Precision Mean | Precision Std | Recall Mean | Recall Std | F1 Mean | F1 Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vas_categorical_baseline | Logistic Regression | 0.716340 | 0.068604 | 0.704505 | 0.105181 | 0.766667 | 0.080027 | 0.727333 | 0.048763 |
| vas_categorical_baseline | KNN | 0.682353 | 0.061112 | 0.678077 | 0.056749 | 0.700000 | 0.230447 | 0.665686 | 0.127232 |
| vas_categorical_baseline | Decision Tree | 0.692157 | 0.089747 | 0.734127 | 0.105065 | 0.580556 | 0.170217 | 0.637863 | 0.136928 |
| vas_categorical_baseline | Random Forest | 0.692157 | 0.089747 | 0.695455 | 0.088268 | 0.675000 | 0.212822 | 0.666687 | 0.141324 |
| vas_categorical_baseline | SVM | 0.670588 | 0.071351 | 0.691190 | 0.047571 | 0.583333 | 0.211321 | 0.613545 | 0.162173 |
| vas_numeric_baseline | Logistic Regression | 0.683007 | 0.081765 | 0.668701 | 0.110190 | 0.744444 | 0.045644 | 0.700000 | 0.057735 |
| vas_numeric_baseline | KNN | 0.590196 | 0.076696 | 0.565501 | 0.073664 | 0.605556 | 0.256400 | 0.566642 | 0.166973 |
| vas_numeric_baseline | Decision Tree | 0.660784 | 0.149693 | 0.647597 | 0.158907 | 0.652778 | 0.205067 | 0.646316 | 0.174300 |
| vas_numeric_baseline | Random Forest | 0.635948 | 0.141839 | 0.598442 | 0.129322 | 0.700000 | 0.230447 | 0.640848 | 0.174434 |
| vas_numeric_baseline | SVM | 0.613725 | 0.081352 | 0.594286 | 0.071856 | 0.605556 | 0.186753 | 0.593684 | 0.130779 |
| womac_candidate | Logistic Regression | 0.671242 | 0.091841 | 0.662987 | 0.115268 | 0.694444 | 0.117851 | 0.673333 | 0.095452 |
| womac_candidate | KNN | 0.681699 | 0.031140 | 0.685934 | 0.093893 | 0.700000 | 0.143533 | 0.678298 | 0.039594 |
| womac_candidate | Decision Tree | 0.601961 | 0.069185 | 0.590043 | 0.074473 | 0.561111 | 0.210085 | 0.561188 | 0.151857 |
| womac_candidate | Random Forest | 0.726144 | 0.054801 | 0.746645 | 0.147314 | 0.716667 | 0.159752 | 0.713571 | 0.070448 |
| womac_candidate | SVM | 0.716993 | 0.064355 | 0.757143 | 0.083163 | 0.630556 | 0.081933 | 0.685000 | 0.066771 |

## Selection under the existing mean-F1 rule

- VAS comparison: `vas_categorical_baseline` with `Logistic Regression` (mean F1 0.727333).
- Baseline vs WOMAC: `vas_categorical_baseline` with `Logistic Regression` (mean F1 0.727333).

## OOF confusion matrices

| Configuration | Model | True Negative | False Positive | False Negative | True Positive | OOF Subjects |
| --- | --- | --- | --- | --- | --- | --- |
| vas_categorical_baseline | Logistic Regression | 30 | 15 | 10 | 33 | 88 |
| vas_categorical_baseline | KNN | 30 | 15 | 13 | 30 | 88 |
| vas_categorical_baseline | Decision Tree | 36 | 9 | 18 | 25 | 88 |
| vas_categorical_baseline | Random Forest | 32 | 13 | 14 | 29 | 88 |
| vas_categorical_baseline | SVM | 34 | 11 | 18 | 25 | 88 |
| vas_numeric_baseline | Logistic Regression | 28 | 17 | 11 | 32 | 88 |
| vas_numeric_baseline | KNN | 26 | 19 | 17 | 26 | 88 |
| vas_numeric_baseline | Decision Tree | 30 | 15 | 15 | 28 | 88 |
| vas_numeric_baseline | Random Forest | 26 | 19 | 13 | 30 | 88 |
| vas_numeric_baseline | SVM | 28 | 17 | 17 | 26 | 88 |
| womac_candidate | Logistic Regression | 29 | 16 | 13 | 30 | 88 |
| womac_candidate | KNN | 30 | 15 | 13 | 30 | 88 |
| womac_candidate | Decision Tree | 29 | 16 | 19 | 24 | 88 |
| womac_candidate | Random Forest | 33 | 12 | 12 | 31 | 88 |
| womac_candidate | SVM | 36 | 9 | 16 | 27 | 88 |

## Per-fold metrics

| Configuration | Model | Fold | Real Training Rows | Synthetic Training Rows | Validation Real Rows | Validation Synthetic Rows | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vas_categorical_baseline | Logistic Regression | 1 | 70 | 500 | 18 | 0 | 0.722222 | 0.750000 | 0.666667 | 0.705882 |
| vas_categorical_baseline | Logistic Regression | 2 | 70 | 500 | 18 | 0 | 0.722222 | 0.700000 | 0.777778 | 0.736842 |
| vas_categorical_baseline | Logistic Regression | 3 | 70 | 500 | 18 | 0 | 0.666667 | 0.615385 | 0.888889 | 0.727273 |
| vas_categorical_baseline | Logistic Regression | 4 | 71 | 500 | 17 | 0 | 0.823529 | 0.857143 | 0.750000 | 0.800000 |
| vas_categorical_baseline | Logistic Regression | 5 | 71 | 500 | 17 | 0 | 0.647059 | 0.600000 | 0.750000 | 0.666667 |
| vas_categorical_baseline | KNN | 1 | 70 | 500 | 18 | 0 | 0.611111 | 0.750000 | 0.333333 | 0.461538 |
| vas_categorical_baseline | KNN | 2 | 70 | 500 | 18 | 0 | 0.722222 | 0.700000 | 0.777778 | 0.736842 |
| vas_categorical_baseline | KNN | 3 | 70 | 500 | 18 | 0 | 0.666667 | 0.615385 | 0.888889 | 0.727273 |
| vas_categorical_baseline | KNN | 4 | 71 | 500 | 17 | 0 | 0.647059 | 0.625000 | 0.625000 | 0.625000 |
| vas_categorical_baseline | KNN | 5 | 71 | 500 | 17 | 0 | 0.764706 | 0.700000 | 0.875000 | 0.777778 |
| vas_categorical_baseline | Decision Tree | 1 | 70 | 500 | 18 | 0 | 0.611111 | 0.750000 | 0.333333 | 0.461538 |
| vas_categorical_baseline | Decision Tree | 2 | 70 | 500 | 18 | 0 | 0.777778 | 0.857143 | 0.666667 | 0.750000 |
| vas_categorical_baseline | Decision Tree | 3 | 70 | 500 | 18 | 0 | 0.777778 | 0.777778 | 0.777778 | 0.777778 |
| vas_categorical_baseline | Decision Tree | 4 | 71 | 500 | 17 | 0 | 0.588235 | 0.571429 | 0.500000 | 0.533333 |
| vas_categorical_baseline | Decision Tree | 5 | 71 | 500 | 17 | 0 | 0.705882 | 0.714286 | 0.625000 | 0.666667 |
| vas_categorical_baseline | Random Forest | 1 | 70 | 500 | 18 | 0 | 0.611111 | 0.750000 | 0.333333 | 0.461538 |
| vas_categorical_baseline | Random Forest | 2 | 70 | 500 | 18 | 0 | 0.777778 | 0.777778 | 0.777778 | 0.777778 |
| vas_categorical_baseline | Random Forest | 3 | 70 | 500 | 18 | 0 | 0.777778 | 0.727273 | 0.888889 | 0.800000 |
| vas_categorical_baseline | Random Forest | 4 | 71 | 500 | 17 | 0 | 0.588235 | 0.555556 | 0.625000 | 0.588235 |
| vas_categorical_baseline | Random Forest | 5 | 71 | 500 | 17 | 0 | 0.705882 | 0.666667 | 0.750000 | 0.705882 |
| vas_categorical_baseline | SVM | 1 | 70 | 500 | 18 | 0 | 0.555556 | 0.666667 | 0.222222 | 0.333333 |
| vas_categorical_baseline | SVM | 2 | 70 | 500 | 18 | 0 | 0.722222 | 0.750000 | 0.666667 | 0.705882 |
| vas_categorical_baseline | SVM | 3 | 70 | 500 | 18 | 0 | 0.722222 | 0.700000 | 0.777778 | 0.736842 |
| vas_categorical_baseline | SVM | 4 | 71 | 500 | 17 | 0 | 0.705882 | 0.714286 | 0.625000 | 0.666667 |
| vas_categorical_baseline | SVM | 5 | 71 | 500 | 17 | 0 | 0.647059 | 0.625000 | 0.625000 | 0.625000 |
| vas_numeric_baseline | Logistic Regression | 1 | 70 | 500 | 18 | 0 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| vas_numeric_baseline | Logistic Regression | 2 | 70 | 500 | 18 | 0 | 0.666667 | 0.636364 | 0.777778 | 0.700000 |
| vas_numeric_baseline | Logistic Regression | 3 | 70 | 500 | 18 | 0 | 0.611111 | 0.583333 | 0.777778 | 0.666667 |
| vas_numeric_baseline | Logistic Regression | 4 | 71 | 500 | 17 | 0 | 0.823529 | 0.857143 | 0.750000 | 0.800000 |
| vas_numeric_baseline | Logistic Regression | 5 | 71 | 500 | 17 | 0 | 0.647059 | 0.600000 | 0.750000 | 0.666667 |
| vas_numeric_baseline | KNN | 1 | 70 | 500 | 18 | 0 | 0.500000 | 0.500000 | 0.222222 | 0.307692 |
| vas_numeric_baseline | KNN | 2 | 70 | 500 | 18 | 0 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| vas_numeric_baseline | KNN | 3 | 70 | 500 | 18 | 0 | 0.666667 | 0.615385 | 0.888889 | 0.727273 |
| vas_numeric_baseline | KNN | 4 | 71 | 500 | 17 | 0 | 0.529412 | 0.500000 | 0.500000 | 0.500000 |
| vas_numeric_baseline | KNN | 5 | 71 | 500 | 17 | 0 | 0.588235 | 0.545455 | 0.750000 | 0.631579 |
| vas_numeric_baseline | Decision Tree | 1 | 70 | 500 | 18 | 0 | 0.444444 | 0.428571 | 0.333333 | 0.375000 |
| vas_numeric_baseline | Decision Tree | 2 | 70 | 500 | 18 | 0 | 0.777778 | 0.727273 | 0.888889 | 0.800000 |
| vas_numeric_baseline | Decision Tree | 3 | 70 | 500 | 18 | 0 | 0.611111 | 0.600000 | 0.666667 | 0.631579 |
| vas_numeric_baseline | Decision Tree | 4 | 71 | 500 | 17 | 0 | 0.647059 | 0.625000 | 0.625000 | 0.625000 |
| vas_numeric_baseline | Decision Tree | 5 | 71 | 500 | 17 | 0 | 0.823529 | 0.857143 | 0.750000 | 0.800000 |
| vas_numeric_baseline | Random Forest | 1 | 70 | 500 | 18 | 0 | 0.444444 | 0.428571 | 0.333333 | 0.375000 |
| vas_numeric_baseline | Random Forest | 2 | 70 | 500 | 18 | 0 | 0.777778 | 0.727273 | 0.888889 | 0.800000 |
| vas_numeric_baseline | Random Forest | 3 | 70 | 500 | 18 | 0 | 0.722222 | 0.700000 | 0.777778 | 0.736842 |
| vas_numeric_baseline | Random Forest | 4 | 71 | 500 | 17 | 0 | 0.529412 | 0.500000 | 0.625000 | 0.555556 |
| vas_numeric_baseline | Random Forest | 5 | 71 | 500 | 17 | 0 | 0.705882 | 0.636364 | 0.875000 | 0.736842 |
| vas_numeric_baseline | SVM | 1 | 70 | 500 | 18 | 0 | 0.500000 | 0.500000 | 0.333333 | 0.400000 |
| vas_numeric_baseline | SVM | 2 | 70 | 500 | 18 | 0 | 0.722222 | 0.700000 | 0.777778 | 0.736842 |
| vas_numeric_baseline | SVM | 3 | 70 | 500 | 18 | 0 | 0.611111 | 0.600000 | 0.666667 | 0.631579 |
| vas_numeric_baseline | SVM | 4 | 71 | 500 | 17 | 0 | 0.588235 | 0.571429 | 0.500000 | 0.533333 |
| vas_numeric_baseline | SVM | 5 | 71 | 500 | 17 | 0 | 0.647059 | 0.600000 | 0.750000 | 0.666667 |
| womac_candidate | Logistic Regression | 1 | 70 | 500 | 18 | 0 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| womac_candidate | Logistic Regression | 2 | 70 | 500 | 18 | 0 | 0.666667 | 0.636364 | 0.777778 | 0.700000 |
| womac_candidate | Logistic Regression | 3 | 70 | 500 | 18 | 0 | 0.611111 | 0.583333 | 0.777778 | 0.666667 |
| womac_candidate | Logistic Regression | 4 | 71 | 500 | 17 | 0 | 0.823529 | 0.857143 | 0.750000 | 0.800000 |
| womac_candidate | Logistic Regression | 5 | 71 | 500 | 17 | 0 | 0.588235 | 0.571429 | 0.500000 | 0.533333 |
| womac_candidate | KNN | 1 | 70 | 500 | 18 | 0 | 0.722222 | 0.833333 | 0.555556 | 0.666667 |
| womac_candidate | KNN | 2 | 70 | 500 | 18 | 0 | 0.666667 | 0.714286 | 0.555556 | 0.625000 |
| womac_candidate | KNN | 3 | 70 | 500 | 18 | 0 | 0.666667 | 0.615385 | 0.888889 | 0.727273 |
| womac_candidate | KNN | 4 | 71 | 500 | 17 | 0 | 0.705882 | 0.666667 | 0.750000 | 0.705882 |
| womac_candidate | KNN | 5 | 71 | 500 | 17 | 0 | 0.647059 | 0.600000 | 0.750000 | 0.666667 |
| womac_candidate | Decision Tree | 1 | 70 | 500 | 18 | 0 | 0.500000 | 0.500000 | 0.222222 | 0.307692 |
| womac_candidate | Decision Tree | 2 | 70 | 500 | 18 | 0 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| womac_candidate | Decision Tree | 3 | 70 | 500 | 18 | 0 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| womac_candidate | Decision Tree | 4 | 71 | 500 | 17 | 0 | 0.588235 | 0.571429 | 0.500000 | 0.533333 |
| womac_candidate | Decision Tree | 5 | 71 | 500 | 17 | 0 | 0.588235 | 0.545455 | 0.750000 | 0.631579 |
| womac_candidate | Random Forest | 1 | 70 | 500 | 18 | 0 | 0.777778 | 1.000000 | 0.555556 | 0.714286 |
| womac_candidate | Random Forest | 2 | 70 | 500 | 18 | 0 | 0.722222 | 0.666667 | 0.888889 | 0.761905 |
| womac_candidate | Random Forest | 3 | 70 | 500 | 18 | 0 | 0.777778 | 0.727273 | 0.888889 | 0.800000 |
| womac_candidate | Random Forest | 4 | 71 | 500 | 17 | 0 | 0.705882 | 0.714286 | 0.625000 | 0.666667 |
| womac_candidate | Random Forest | 5 | 71 | 500 | 17 | 0 | 0.647059 | 0.625000 | 0.625000 | 0.625000 |
| womac_candidate | SVM | 1 | 70 | 500 | 18 | 0 | 0.722222 | 0.833333 | 0.555556 | 0.666667 |
| womac_candidate | SVM | 2 | 70 | 500 | 18 | 0 | 0.666667 | 0.714286 | 0.555556 | 0.625000 |
| womac_candidate | SVM | 3 | 70 | 500 | 18 | 0 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| womac_candidate | SVM | 4 | 71 | 500 | 17 | 0 | 0.705882 | 0.714286 | 0.625000 | 0.666667 |
| womac_candidate | SVM | 5 | 71 | 500 | 17 | 0 | 0.823529 | 0.857143 | 0.750000 | 0.800000 |

## Reproduction command

```powershell
python ml\run_controlled_experiments.py
```

The complete subject-level OOF predictions and probabilities are stored in `oof_predictions.csv`; the shared real-only validation assignments are stored in `fold_assignments.csv`; and the full machine-readable record is stored in `controlled_experiment_results.json`.
