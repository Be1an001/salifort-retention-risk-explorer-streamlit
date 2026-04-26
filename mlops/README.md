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

Phase 6 adds an optional Airflow DAG scaffold under `orchestration/airflow/dags/`. It orchestrates the existing MLOps CLI scripts for local/dev review only; Streamlit does not trigger it, and it does not write to `artifacts/v2/`.

Phase 7 adds GitHub Actions CI checks for app compile safety, MLOps contract tests, static Airflow DAG validation, and Docker Compose configuration. CI does not deploy, publish images, or require generated lab artifacts.

Phase 8 adds a read-only Streamlit MLOps Lab page. The page displays local lab status and documentation context but does not execute training, Docker, MLflow, Airflow, git, CI, or background jobs.

Hosted Streamlit mode adds an Online CSV Insight sandbox inside the MLOps Lab page. It works without a deployed FastAPI backend: uploads are processed in memory, a transparent pandas heuristic creates a review-priority queue, identifier-like fields are excluded from summaries, and only compact aggregate statistics are sent to OpenAI for optional briefings.

Local/dev FastAPI remains part of the MLOps showcase for technical review, but it is not required for the hosted Streamlit CSV Insight workflow.

The committed MLOps Evidence Pack under `docs/demo-assets/mlops-evidence/` summarizes selected local/dev outputs for online reviewers. It is sanitized, lightweight, and excludes joblib model files, `mlruns/`, uploaded CSVs, secrets, and local absolute paths.

Hosted Streamlit packaged model inference uses a separate exported artifact under `artifacts/mlops_lab_online/`. That artifact is created from the local/dev lab champion for online demonstration only and still does not replace the public weighted XGBoost threshold `0.29` app truth.
