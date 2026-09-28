# Dataset Overview

This project uses a compact clinical dataset for knee osteoarthritis (KOA) risk-marker screening research. The prototype is designed for early risk assessment and research use, not as a clinical diagnostic system.

## Dataset summary

- Total subjects: 88
- Healthy cases: 45
- Knee OA cases: 43
- Target variable: `group`
- `group = 0`: healthy
- `group = 1`: knee osteoarthritis

## Features used in the current model

The selected model uses the following predictors:

- `age`
- `gender`
- `BMI`
- `VAS score`
- `JPR_30`
- `JPR_45`
- `JPR_60`

These features align with the current model training configuration and the saved feature list in `models/features.pkl`.

## Why K-L grade and KOA period are not used as predictors

The training workflow intentionally excludes `K-L grade` and `KOA period` from the feature set because they can contain disease information that leaks target-relevant knowledge into the model. This would make the model appear stronger than it is and would reduce the value of a real screening use case. For a prototype research system, the goal is to evaluate whether risk can be estimated from clinically observable and accessible screening inputs rather than from diagnosis-derived labels.

## Limitations

- The dataset is small for robust clinical generalization.
- It reflects a prototype research sample rather than the full population of the North Eastern Region.
- The model should be viewed as an AI-assisted screening prototype, not a validated clinical decision tool.
- Results should be interpreted as research-oriented risk assessment and not medical diagnosis.

## Source and usage

This repository preserves the project’s current dataset source and uses the available local CSV file without altering the model contract or core training logic. If a formal source citation or licensing notice exists in the project materials, it should be respected alongside this summary.
