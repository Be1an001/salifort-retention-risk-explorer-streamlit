# MLOps Mini-Lab Demo Guide

This guide shows how to demonstrate the Salifort MLOps Mini-Lab without implying that it is production HR infrastructure.

The project has three demo paths:

- **Hosted Streamlit path:** use the MLOps Lab Online CSV Insight sandbox directly in the hosted app.
- **Local/dev MLOps path:** run the CLI pipeline, local FastAPI service, Docker Compose stack, MLflow UI, and Airflow DAG validator on a development machine.
- **GitHub/CI evidence path:** review the GitHub Actions workflow and tests that validate app runtime, MLOps contracts, Airflow static checks, and Docker Compose configuration.

## Hosted Streamlit Demo

Use this path for the fastest portfolio review.

1. Open the hosted Streamlit app.
2. Go to **MLOps Lab**.
3. Open **Online CSV Insight**.
4. Download the **100-row synthetic demo CSV**.
5. Upload the same CSV back into the sandbox.
6. Review the deterministic insight pack:
   - High / Medium / Low review-band counts
   - priority rows by `uploaded_row_id`
   - department review summary
   - top departments by high-review count and high-review rate
   - most common review drivers
7. Optionally generate the OpenAI briefing if `OPENAI_API_KEY` is configured.
8. Download the review summary CSV.

The hosted sandbox uses a transparent pandas heuristic. It does not call FastAPI, Docker, MLflow, Airflow, joblib model artifacts, or training scripts.

## Local MLOps Pipeline Demo

From the repository root:

```bash
python scripts/mlops_run_pipeline.py
```

This local/dev command prepares data, trains lab candidate models, evaluates them, logs MLflow runs, and writes gitignored lab outputs:

- `mlops/data/processed/train.parquet`
- `mlops/data/processed/test.parquet`
- `mlops/reports/data_profile.json`
- `mlops/reports/training_results.json`
- `mlops/reports/evaluation_summary.json`
- `mlops/reports/model_card.md`
- `mlops/models/champion_model.joblib`

These files are local lab artifacts. They are intentionally not committed and do not replace `artifacts/v2`.

## Export Evidence Pack

After running the local pipeline, export a sanitized, lightweight evidence snapshot:

```bash
python scripts/export_mlops_evidence_pack.py
```

The evidence pack is written to `docs/demo-assets/mlops-evidence/` and can be inspected in the hosted **MLOps Evidence** tab. It includes pipeline, training/evaluation, FastAPI, Docker Compose, Airflow, and GitHub Actions summaries without copying model binaries, `mlruns/`, secrets, uploaded CSVs, or local absolute paths.

## FastAPI Local Demo

After running the local pipeline, start the optional API:

```bash
python -m uvicorn api.main:app --reload
```

Useful local URLs:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/model-info`
- `http://127.0.0.1:8000/docs`

The FastAPI service is local/dev evidence for model serving. The hosted Streamlit app does not require it.

## Docker Compose Demo

Validate or run the optional local stack:

```bash
docker compose config
docker compose up api
docker compose up streamlit
docker compose down
```

Docker Compose is for local demonstration only. It does not change the public Streamlit model truth.

## MLflow Demo

After running the local pipeline:

```bash
mlflow ui
```

MLflow runs are local and gitignored under `mlruns/`.

## Airflow DAG Evidence

The repo includes a local/dev DAG scaffold:

```bash
python scripts/validate_mlops_airflow_dag.py
```

DAG ID:

```text
salifort_mlops_mini_lab_pipeline
```

The DAG orchestrates lab CLI scripts only. It is not triggered by Streamlit and does not write `artifacts/v2`.

## GitHub Actions / CI

The CI workflow validates:

- app runtime import/compile checks
- MLOps package and API tests
- Airflow DAG static validation without installing Airflow
- Docker Compose config checks

CI does not deploy, publish Docker images, upload model artifacts, or run production workflows.

## What This Project Does Not Claim

- It is not a production HR system.
- It is not an employment decision system.
- It does not autonomously retrain or deploy models.
- It does not host Airflow or MLflow as production services.
- It does not replace the public weighted XGBoost reference model or selected threshold `0.29`.
- It does not send raw uploaded CSV rows or identifier-like fields to OpenAI.
