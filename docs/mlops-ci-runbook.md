# Salifort MLOps Mini-Lab CI Runbook

## Purpose

The GitHub Actions workflow validates the Streamlit app runtime and the optional MLOps Mini-Lab extension without deploying anything or changing public model truth.

Workflow file:

```text
.github/workflows/ci.yml
```

## What CI Checks

The `app-runtime-checks` job:

- installs `requirements.txt`
- compiles the Streamlit entry point, data loader, pages, services, and view models
- does not start a Streamlit server
- does not require OpenAI keys or retrieval runtime calls

The `mlops-tests` job:

- installs `requirements-mlops.txt`
- compiles `src/salifort_mlops/`, `api/`, and the Airflow DAG scaffold
- runs `scripts/validate_mlops_airflow_dag.py`
- runs MLOps data, feature, evaluation, training-helper, API, and Airflow contract tests

The `docker-config-check` job:

- runs `docker compose config`
- runs `docker compose --profile mlflow config`
- does not run containers
- does not publish images

## What CI Intentionally Does Not Do

- does not deploy the Streamlit app
- does not publish Docker images
- does not push generated artifacts
- does not upload lab model files
- does not run Airflow scheduler or webserver
- does not install `apache-airflow`
- does not run production workflows
- does not require `references/source_workflow/`
- does not require `mlruns/`
- does not require committed generated model artifacts
- does not modify `artifacts/v2/`

## Airflow Validation Boundary

Airflow is statically validated only. The validator compiles the DAG file and checks the DAG ID, task IDs, dependency chain, and forbidden action patterns. If `apache-airflow` is unavailable, the optional DAG import is skipped and the static checks still run.

This keeps CI lightweight and avoids turning the portfolio repo into an Airflow deployment.

## Generated Artifact Boundary

The MLOps lab can generate parquet datasets, joblib models, reports, and MLflow runs locally. Those outputs remain gitignored and are not required for CI. API prediction tests skip model-positive checks when no lab champion model is present, while still validating controlled missing-model behavior.

## Docker Boundary

CI validates Compose syntax for the default stack and the optional MLflow profile. Docker image builds remain a local validation step because the MLOps API image includes modeling dependencies and can be slow for a portfolio CI run.

## Local Commands

Run the MLOps regression suite:

```powershell
python -m pytest tests/test_data_contract.py tests/test_feature_pipeline.py tests/test_evaluation_metrics.py tests/test_training_pipeline.py tests/test_api_contract.py tests/test_prediction_service.py tests/test_airflow_dag_contract.py
```

Run Airflow static validation:

```powershell
python scripts/validate_mlops_airflow_dag.py
```

Run Docker Compose config checks:

```powershell
docker compose config
docker compose --profile mlflow config
```

Optional local Docker image build:

```powershell
docker compose build api
docker compose build streamlit
```

## Project Boundary

CI supports the MLOps Mini-Lab as a local/dev portfolio extension. It does not make the existing Streamlit app depend on FastAPI, Docker, MLflow, or Airflow, and it does not alter the public weighted XGBoost threshold `0.29` story.
