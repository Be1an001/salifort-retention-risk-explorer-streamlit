# Generated Artifact Guide

This directory stores the generated files used by the Streamlit app.

The app should continue to run when these files are absent. Streamlit runtime should only read them when they are present. It should not regenerate model outputs, infer values from images, or retrain models.

## Runtime policy

- Fallback behavior remains active when `artifacts/v2/` is empty.
- The app may selectively load generated files when they are available.
- Artifact generation should happen outside Streamlit runtime in an offline modeling or packaging workflow.

## Stable row-key strategy

Row-level generated files should use `employee_id_v2` as the primary join key.

`employee_id_v2` is defined in the app loader as a deterministic hash of the cleaned row content using these canonical fields:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_monthly_hours`
- `tenure`
- `work_accident`
- `left`
- `promotion_last_5years`
- `department`
- `salary`

This avoids relying on row order and is safer for joining `employee_scores.parquet` outputs back into the app.

## Required artifacts

- `employee_scores.parquet`
- `department_exposure.csv`
- `threshold_curve.csv`
- `validation_model_comparison.csv`
- `confusion_matrix_at_selected_threshold.csv`
- `shap_importance.csv`
- `metadata.json`

## Optional artifacts

- `employee_shap_sample.parquet`
- `pr_curve_points.parquet`
- `model_modes_summary.json`

## Contract files in this directory

- `metadata.template.json`: starter template for the required metadata artifact
- `schemas/artifact_contract.json`: machine-friendly schema contract for required and optional generated files

## Offline builder

The first offline builder for this repo lives at:

- `scripts/build_v2_artifacts.py`

It is designed to run outside Streamlit runtime and uses the trusted sibling modeling workflow as the source-of-truth reference:

- `../salifort-motors-attrition-modeling-python/scripts/02_salifort_motors_capstone_portfolio_project.py`

The builder uses this repo's checked-in CSV at `data/hr_capstone_dataset.csv`, mirrors the cleaned-column conventions already used by the app, and reuses the deterministic `employee_id_v2` row-key strategy from `app/utils/load_data.py`.

For `--model-mode operational`, the builder preserves the public operational source-of-truth framing from the sibling project repo: the published selection is weighted XGBoost at threshold `0.29`. The builder still writes the full rerun validation comparison table so any local metric drift remains visible.

## How to run

From the repo root:

```powershell
python scripts/build_v2_artifacts.py
```

Optional flags:

```powershell
python scripts/build_v2_artifacts.py --dry-run
python scripts/build_v2_artifacts.py --skip-shap
python scripts/build_v2_artifacts.py --model-mode operational
python scripts/build_v2_artifacts.py --model-mode survey_rich
```

## What it generates

When the trusted source workflow and required local ML dependencies are available, the builder can generate:

- `metadata.json`
- `employee_scores.parquet`
- `department_exposure.csv`
- `threshold_curve.csv`
- `validation_model_comparison.csv`
- `confusion_matrix_at_selected_threshold.csv`
- `shap_importance.csv`
- `pr_curve_points.parquet`
- `model_modes_summary.json`

It also attempts `employee_shap_sample.parquet` when SHAP is available.

## Dependency note

The Streamlit runtime requirements were intentionally left unchanged in this repo.

The offline builder depends on the modeling stack used by the trusted source workflow, including:

- `scikit-learn`
- `xgboost`
- `imbalanced-learn`
- `shap`

If those packages are not available locally, the builder exits with a short dependency message instead of generating partial fake outputs.
