# Salifort MLOps Mini-Lab

This folder supports the local/dev MLOps Mini-Lab extension for the Salifort Motors Retention Risk Explorer.

It is separate from `artifacts/v2/` and does not replace the public Streamlit app story:

- public reference model: Weighted XGBoost
- selected public threshold: `0.29`
- public app runtime: artifact-backed Streamlit pages
- MLOps Lab packaged demo threshold: `0.60` in `artifacts/mlops_lab_online/model_metadata.json`

## Current Role

Local/dev MLOps commands can write processed lab datasets, trained lab models, reports, and MLflow runs to this area and to `mlruns/`. Those outputs are for local development and portfolio demonstration only. They are intentionally gitignored and should not be committed.

The hosted Streamlit app does not depend on generated files in this folder to open. Streamlit should not trigger training, Docker, Airflow, MLflow, FastAPI, git, CI, or shell workflows.

## Local/Dev Components

- **Pipeline:** `python scripts/mlops_run_pipeline.py` prepares data, trains lab-only candidates, evaluates them, logs MLflow runs, and writes local reports.
- **MLflow:** `mlflow ui` can inspect local runs under `mlruns/`.
- **FastAPI:** `python -m uvicorn api.main:app --reload` serves the local lab champion when `mlops/models/champion_model.joblib` exists and returns controlled missing-model messages otherwise.
- **Docker Compose:** `docker compose config`, `docker compose up api`, `docker compose up streamlit`, and `docker compose --profile mlflow up mlflow` validate or run local demo services.
- **Airflow DAG scaffold:** `python scripts/validate_mlops_airflow_dag.py` validates the local/dev DAG contract under `orchestration/airflow/dags/`.
- **CI:** GitHub Actions compiles app/MLOps files, runs contract tests, validates the Airflow DAG statically, and checks Docker Compose config. CI does not deploy or publish artifacts.

## Hosted Streamlit MLOps Lab

Hosted Streamlit mode includes an Online CSV Insight sandbox inside the MLOps Lab page. It works without a deployed FastAPI backend:

- uploads are processed in memory
- transparent pandas heuristic scoring creates a review-priority queue
- packaged demo model inference scores rows from `artifacts/mlops_lab_online/`
- identifier-like fields are excluded from displayed/downloaded summaries
- optional OpenAI briefings use compact aggregate JSON only
- raw CSV rows and PII are not sent to OpenAI

Local/dev FastAPI remains useful for technical review, but it is not required for the hosted Streamlit CSV Insight workflow.

## Evidence and Exports

The committed MLOps Evidence Pack under `docs/demo-assets/mlops-evidence/` summarizes selected local/dev outputs for online reviewers. It is sanitized, lightweight, and excludes joblib model files, `mlruns/`, uploaded CSVs, secrets, and local absolute paths.

The hosted packaged model artifact under `artifacts/mlops_lab_online/` is created from the local/dev lab champion with:

```bash
python scripts/export_streamlit_model_artifact.py
```

That artifact enables hosted Streamlit demo inference only. It still does not replace the public Weighted XGBoost threshold `0.29` app truth.

## Boundaries

- Not a production HR system.
- Not an employment decision system.
- Not a hosted FastAPI, Docker, MLflow, or Airflow deployment.
- Not a mechanism for updating `artifacts/v2/` from Streamlit.
- Human review support and technical portfolio evidence only.
