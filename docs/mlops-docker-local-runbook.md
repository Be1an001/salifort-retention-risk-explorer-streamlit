# Salifort MLOps Mini-Lab Docker Local Runbook

## Purpose

This runbook explains how to run the optional local/dev Docker demo for the Salifort MLOps Mini-Lab.

The Docker stack is separate from the public Streamlit app truth. It does not change `artifacts/v2/`, does not change the public weighted XGBoost threshold `0.29` story, and is not a production HR deployment.

## Prerequisites

- Docker Desktop is installed and running.
- Run commands from the repository root.
- Optional: run the MLOps pipeline first if API predictions should work:

```powershell
python scripts/mlops_run_pipeline.py
```

If the lab model is missing, the API still starts. `/health` and `/model-info` return setup guidance, while `/predict` and `/batch-predict` return a controlled `503` response.

## Build Images

```powershell
docker compose build
```

Build individual services:

```powershell
docker compose build api
docker compose build streamlit
```

## Run Services

Run the API service:

```powershell
docker compose up api
```

Run the Streamlit service:

```powershell
docker compose up streamlit
```

Run the default stack:

```powershell
docker compose up
```

Run the optional MLflow UI:

```powershell
docker compose --profile mlflow up mlflow
```

Stop and remove containers:

```powershell
docker compose down
```

## Local URLs

- FastAPI docs: http://localhost:8000/docs
- FastAPI health: http://localhost:8000/health
- Streamlit app: http://localhost:8501
- MLflow UI: http://localhost:5000

## Mounted Lab Artifacts

The API image does not bake generated lab models into the container. Compose mounts local folders instead:

- `./mlops:/app/mlops:ro`
- `./data:/app/data:ro`

The API looks for:

- model: `/app/mlops/models/champion_model.joblib`
- metadata: `/app/mlops/reports/evaluation_summary.json`

Generated models, reports, parquet files, and `mlruns/` remain gitignored.

## Service Boundaries

- The existing Streamlit app remains artifact-backed.
- Streamlit does not require FastAPI, Docker, MLflow, or Airflow to open outside this Docker demo.
- Streamlit does not trigger training, MLflow, Docker, or Airflow.
- The FastAPI model is a local/dev MLOps lab model only.
- Predictions are portfolio demonstration and human-review support only, not employment decisions.
- Airflow is not included in this phase.

## Troubleshooting

Check Docker:

```powershell
docker --version
docker compose version
docker info
```

Validate Compose:

```powershell
docker compose config
```

Check API health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

If API predictions return `503`, run:

```powershell
python scripts/mlops_run_pipeline.py
docker compose up api
```
