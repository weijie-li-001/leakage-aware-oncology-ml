# Leakage-aware treatment-pattern classification

This repository is a privacy-safe demonstration of a retrospective machine-learning workflow for classifying historical oncology treatment patterns. It is an analytical example, not a treatment recommendation system.

## Privacy statement

No original workbook, patient-level record, company document, analysis export, trained model, or result derived from the private source data is included. The public example uses deterministic synthetic data generated in code. See [DATA_PRIVACY.md](DATA_PRIVACY.md) before adding files or publishing changes.

## What the pipeline demonstrates

- Separating the treatment decision from regimen-family classification.
- Excluding identifiers, outcomes, treatment fields, and study membership from predictors.
- Fitting imputation, scaling, and one-hot encoding inside each validation fold.
- Nested stratified cross-validation for model selection and out-of-fold evaluation.
- Leave-one-group-out validation to make cohort shift visible.
- Reporting macro-F1 and balanced accuracy instead of relying on accuracy alone.

Regimen-family mappings are transparent heuristics for software demonstration only. They have not been clinically validated.

## Run the synthetic demo

```bash
python -m pip install -r requirements.txt
python generate_synthetic_data.py --output synthetic_demo.xlsx
python chemo_regimen_pipeline.py --data synthetic_demo.xlsx --output artifacts
python -m unittest -v
```

The generated workbook and all analysis artifacts are ignored by Git. The full evaluation compares several models and can take a few minutes.

## Repository contents

- `chemo_regimen_pipeline.py` - data preparation, leakage controls, evaluation, and plots.
- `generate_synthetic_data.py` - creates a deterministic, entirely fictional workbook for demonstration.
- `test_chemo_regimen_pipeline.py` - tests label semantics and feature-exclusion policy using only synthetic records.
- `DATA_PRIVACY.md` - rules for keeping private or derived data out of version control.

## Intended use and limitations

The target is a historical decision, not a counterfactual clinical outcome. The code cannot determine which treatment would be best for a patient, and it must not be used for medical decisions or deployment. Validation results produced from the synthetic demo are illustrative and have no clinical meaning.
