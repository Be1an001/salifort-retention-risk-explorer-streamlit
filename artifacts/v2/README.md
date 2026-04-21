# V2 Generated Artifacts

This folder stores generated files that the Streamlit app can read at runtime.

The app should only load these files. It should not retrain models, regenerate SHAP values, infer values from images, or silently invent missing outputs while a visitor is using the app.

## Runtime Rule

- If the required generated files are present, the app uses them for supported pages.
- If row-level generated files are missing or incomplete, the app keeps running with a simpler fallback screening view.
- The fallback view is for exploration only. It is not the final weighted XGBoost probability.
- Artifact generation belongs in an offline script, not in the Streamlit runtime.

## Required Files

- `metadata.json`
- `employee_scores.parquet`
- `department_exposure.csv`
- `threshold_curve.csv`
- `validation_model_comparison.csv`
- `confusion_matrix_at_selected_threshold.csv`
- `shap_importance.csv`

## Optional Files

- `employee_shap_sample.parquet`
- `pr_curve_points.parquet`
- `model_modes_summary.json`

## Contract and Template Files

- `metadata.template.json`: starter shape for the metadata artifact.
- `schemas/artifact_contract.json`: machine-readable contract for required and optional artifacts.

## Stable Employee Key

Row-level generated files should use `employee_id_v2` as the join key.

`employee_id_v2` is a deterministic hash built from cleaned row values such as satisfaction, evaluation, projects, monthly hours, tenure, accident history, outcome, promotion history, department, and salary. This is safer than relying on row order.

## Offline Builder

The offline builder is:

```powershell
python scripts/build_v2_artifacts.py
```

Useful options:

```powershell
python scripts/build_v2_artifacts.py --dry-run
python scripts/build_v2_artifacts.py --skip-shap
python scripts/build_v2_artifacts.py --model-mode operational
python scripts/build_v2_artifacts.py --model-mode survey_rich
```

The public portfolio app preserves the operational reference story: weighted XGBoost at threshold `0.29`. The builder can still write comparison tables so local metric differences remain visible.

## Extra Local Dependencies

The Streamlit runtime requirements stay lightweight. The offline builder may also need modeling packages used by the original workflow, such as:

- `scikit-learn`
- `xgboost`
- `imbalanced-learn`
- `shap`

If those packages are missing, the builder should stop with a clear dependency message instead of writing partial fake outputs.
