# MLOps Lab Online Demo Model

This folder contains a small packaged model artifact for Streamlit-native MLOps Lab inference.

It is separate from `artifacts/v2/` and does **not** replace the public app model truth:

- public reference model: weighted XGBoost
- public selected threshold: `0.29`

Files:

- `champion_model.joblib`: exported local/dev MLOps Mini-Lab champion pipeline for hosted Streamlit demo inference.
- `model_metadata.json`: sanitized model metadata, threshold, metrics, required input columns, and responsible-use boundaries.

Regenerate from local/dev lab outputs:

```bash
python scripts/mlops_run_pipeline.py
python scripts/export_streamlit_model_artifact.py
```

This artifact is for portfolio demonstration and human review support only. It is not a production HR model and not an employment decision system.
