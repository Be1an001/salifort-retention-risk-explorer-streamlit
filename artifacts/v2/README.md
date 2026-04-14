# V2 Precomputed Artifact Contract

This directory is reserved for future precomputed V2 artifacts used by the Streamlit app.

The current deployed app must continue to run when these files are absent. Streamlit runtime should only read these artifacts when they are present. It should not regenerate model outputs, infer values from images, or retrain models.

## Runtime policy

- Current behavior: V1 fallback remains active when `artifacts/v2/` is empty.
- Future behavior: the app may selectively load precomputed V2 artifacts when they are available.
- Artifact generation should happen outside Streamlit runtime in an offline modeling or packaging workflow.

## Stable row-key strategy

Future row-level V2 artifacts should use `employee_id_v2` as the primary join key.

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

This avoids relying on row order and is safer for joining future `employee_scores.parquet` outputs back into the app.

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
- `schemas/artifact_contract.json`: machine-friendly schema contract for required and optional V2 artifacts
