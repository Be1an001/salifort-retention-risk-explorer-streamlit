# Salifort MLOps Mini-Lab Airflow Local Runbook

## Purpose

This runbook explains the optional local/dev Airflow DAG scaffold for the Salifort MLOps Mini-Lab.

The DAG orchestrates existing CLI scripts for the lab pipeline. It does not run inside Streamlit, does not update `artifacts/v2/`, and does not replace the public weighted XGBoost threshold `0.29` story.

## Boundary

- Local/dev orchestration only.
- No production scheduler is configured in this phase.
- No automatic model deployment.
- No HR alerts.
- No Streamlit-triggered execution.
- Generated lab models, reports, parquet files, and MLflow runs remain local and gitignored.

## DAG File

```text
orchestration/airflow/dags/salifort_mlops_pipeline.py
```

DAG ID:

```text
salifort_mlops_mini_lab_pipeline
```

Tasks:

- `prepare_data`: runs `python scripts/mlops_01_prepare_data.py`
- `train_model`: runs `python scripts/mlops_02_train_model.py`
- `evaluate_model`: runs `python scripts/mlops_03_evaluate_model.py`
- `validate_api_contract`: runs selected API contract tests

Dependency order:

```text
prepare_data >> train_model >> evaluate_model >> validate_api_contract
```

Review-only PACE mapping:

- Analyze: `prepare_data`
- Construct: `train_model` and `evaluate_model`
- Execute: `validate_api_contract` and package reviewer evidence outside the DAG

This mapping is an explanatory portfolio frame, not an official Google PACE automation product.

## Environment Variable

Set `SALIFORT_PROJECT_ROOT` to the repository root in your local Airflow environment:

```bash
export SALIFORT_PROJECT_ROOT=/path/to/salifort-retention-risk-explorer-streamlit
```

PowerShell equivalent:

```powershell
$env:SALIFORT_PROJECT_ROOT="C:\path\to\salifort-retention-risk-explorer-streamlit"
```

If the variable is not set, the DAG uses a repository-relative fallback based on its file location.

## How to Use

Use one of these approaches:

- Copy or mount `orchestration/airflow/dags/salifort_mlops_pipeline.py` into an existing local Airflow instance.
- Point a local Airflow DAG folder at `orchestration/airflow/dags/`.
- Treat the DAG as a reviewed scaffold when Airflow is not installed.

This phase intentionally does not install Airflow, add Airflow to project requirements, or require a running scheduler.

## Manual Validation

Run static DAG validation:

```powershell
python scripts/validate_mlops_airflow_dag.py
```

Run the DAG contract test:

```powershell
python -m pytest tests/test_airflow_dag_contract.py
```

Run the broader local regression suite:

```powershell
python -m pytest tests/test_data_contract.py tests/test_feature_pipeline.py tests/test_evaluation_metrics.py tests/test_training_pipeline.py tests/test_api_contract.py tests/test_prediction_service.py tests/test_airflow_dag_contract.py
```

The validation script compiles the DAG and checks the expected DAG ID, task IDs, dependency order, and blocked action patterns. If `apache-airflow` is not installed, it skips the optional runtime import and still validates the static contract.

## Known Limitations

- The BashOperator commands assume a Linux-like Airflow runtime.
- The validation task requires local test dependencies to be installed.
- The DAG is not a deployment mechanism.
- The DAG does not include Dockerized Airflow in this phase.
- The DAG does not make the Streamlit app depend on Airflow.
