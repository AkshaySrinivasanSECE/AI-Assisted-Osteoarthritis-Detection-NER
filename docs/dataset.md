# Dataset and provenance

## Dataset identity

- Preserved observed subjects: 88.
- Observed target distribution: 45 healthy and 43 knee OA.
- Generated training examples: 500, balanced as 250 per target class.
- Active training rows: 588.

The authoritative counts and distribution summaries are written to `data/synthetic_data_quality_report.json` and `models/model_metadata.json`.

`data/full_indicators_summary.original.csv` is the preserved observed source. `data/full_indicators_summary.csv` is generated and may be recreated; it is the unified 588-row modeling table, not 588 observed patients.

## Model fields

The target is `group`, where `0` represents healthy and `1` represents knee osteoarthritis.

The model uses only:

- `age`
- `gender`
- `BMI`
- `VAS score`

Identifiers, K-L grade, KOA period, WOMAC, JPR, EMG, provenance, and every other source column are excluded from the classifier. Height and weight are frontend inputs used only to calculate BMI. Supporting symptoms and history are not model features.

## Synthetic generation

The generator validates the original data and uses SMOTENC with the recorded fixed seed. Gender and VAS score are categorical for sampling. Generated candidates are rounded to the expected data types, checked against allowed ranges, filtered to remove copied real feature vectors and duplicate synthetic feature vectors, and reduced to the exact balanced target counts.

Every active row receives:

- a unique `record_id`
- `record_source` equal to `real` or `synthetic`
- `generation_method` equal to `observed` or `SMOTENC`
- `validation_eligible` equal to true for every active row; `record_source` is retained for audit only

Regeneration starts from the preserved source and replaces the active table; it does not append to a previous generated dataset.

## Validation boundary

All active rows are split into outer validation folds and are modeled together. The 500 synthetic examples are already present in the active table; they are not regenerated inside folds or excluded from validation and calibration. They still do not represent additional patients or additional clinical evidence.

## Quality report

The machine-readable quality report compares real and synthetic age, gender, BMI, and VAS-score distributions overall and by target. It also records validation results, duplicate filtering, generation configuration, hashes, and the explicit claim guardrail.

## Limitations

- The original sample is small.
- Synthetic rows reflect patterns available in the original data and cannot add new clinical information.
- Categorical imbalance in the observed sample can carry into generated data.
- No external cohort has been used to establish generalizability.
- The data supports an educational screening prototype, not clinical validation or diagnosis.
