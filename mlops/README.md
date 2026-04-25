# Salifort MLOps Mini-Lab

This folder is reserved for the local/dev MLOps Mini-Lab extension.

It is separate from `artifacts/v2/` and does not replace the public Streamlit app story:

- public reference model: weighted XGBoost
- selected public threshold: `0.29`
- current app runtime: artifact-backed Streamlit pages

Future phases may write processed lab datasets, trained lab models, and report files here. Those outputs are for local development and portfolio demonstration only.

The existing Streamlit pages should not depend on this folder to open, and Streamlit should not trigger training, Docker, Airflow, MLflow, or FastAPI workflows.

Phase 3 adds local training and MLflow tracking for lab-only candidate models. Those runs write to `mlruns/`, `mlops/models/`, and `mlops/reports/`; they are intentionally ignored by git and do not update `artifacts/v2/` or the public weighted XGBoost threshold `0.29` story.

Phase 4 adds an optional FastAPI serving layer under `api/`. The service loads the lab champion model from `mlops/models/` when available, returns controlled missing-model messages when it is not available, and remains separate from the existing Streamlit runtime.

Phase 5 adds optional Docker Compose infrastructure for local demos. Compose mounts generated lab artifacts from this folder instead of committing or baking them into images. The existing Streamlit app still runs independently with `streamlit run app/app.py`.
