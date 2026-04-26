# MLOps Evidence Pack

This folder contains sanitized, lightweight evidence snapshots for the Salifort MLOps Mini-Lab.

It is intended for online reviewers who cannot run the local/dev pipeline on the user's computer.

Included evidence:

- `pipeline_run_summary.json`: data-prep and split summary.
- `training_evaluation_summary.json`: candidate model, lab champion, metric, and MLflow tracking summary.
- `fastapi_health_example.json`: sanitized `/health` response shape.
- `fastapi_model_info_example.json`: sanitized `/model-info` response shape with repo-relative artifact paths.
- `docker_compose_validation.md`: local/dev Docker Compose validation summary.
- `airflow_validation_summary.md`: local/dev DAG validation summary.
- `github_actions_summary.md`: CI validation summary.

Excluded by design:

- joblib model files
- `mlruns/`
- uploaded CSV files
- API keys, `.env`, or `.streamlit/secrets.toml`
- large generated data files
- local absolute paths

Boundary: Local/dev MLOps Mini-Lab evidence only; not production HR and not an employment decision system.

Public app truth: Public app truth remains weighted XGBoost at threshold 0.29.

Regenerate locally after running the MLOps pipeline:

```bash
python scripts/mlops_run_pipeline.py
python scripts/export_mlops_evidence_pack.py
```
