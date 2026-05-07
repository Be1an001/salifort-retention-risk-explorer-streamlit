---
name: mlops-validation
description: Validate Salifort local/dev MLOps surfaces and evidence boundaries. Use when asked to check MLOps Lab, FastAPI, Docker Compose, MLflow docs, Airflow DAG scaffolds, CI, or evidence-pack consistency without retraining.
---

# MLOps Validation Skill

## Inputs to Inspect

- `app/pages/mlops_lab.py`
- `docs/mlops-demo-guide.md`
- `docs/mlops-docker-local-runbook.md`
- `docs/mlops-airflow-local-runbook.md`
- `docs/mlops-ci-runbook.md`
- `docs/demo-assets/mlops-evidence/`
- `artifacts/mlops_lab_online/README.md`
- `artifacts/mlops_lab_online/model_metadata.json`
- `api/`
- `src/salifort_mlops/`
- `scripts/mlops_*.py`
- `scripts/validate_mlops_airflow_dag.py`
- `docker-compose.yml`
- `orchestration/airflow/dags/`
- `.github/workflows/ci.yml`
- `tests/test_mlops_evidence_pack_contract.py`
- `tests/test_online_csv_insight_contract.py`

## Steps

1. Confirm the task is validation/review unless the user explicitly requests changes.
2. Verify hosted Streamlit behavior is separate from local/dev MLOps.
3. Verify packaged model metadata identifies `model_scope` as `mlops-lab-online-demo`.
4. Verify public app threshold `0.29` remains separate from lab threshold `0.60`.
5. Check that generated local outputs remain ignored and unstaged.
6. Run only safe validation commands unless the user asks for services or training.
7. Report local/dev evidence as demo evidence, not production MLOps.

## Boundaries

- Do not run `scripts/mlops_run_pipeline.py` unless explicitly requested.
- Do not run export scripts unless explicitly requested.
- Do not start FastAPI, Streamlit, MLflow, Docker services, or Airflow unless explicitly requested.
- Do not modify `mlops/data`, `mlops/models`, `mlops/reports`, `mlruns`, or model artifacts during validation.
- Do not claim CI deploys or publishes models.

## Expected Outputs

- validation summary
- command results
- staged-file safety check when committing
- clear hosted vs local/dev explanation

## Done Criteria

- Safe commands complete or failures are explained.
- No generated outputs, model binaries, caches, or secrets are staged.
- MLOps Lab wording remains local/dev and portfolio-demo scoped.
