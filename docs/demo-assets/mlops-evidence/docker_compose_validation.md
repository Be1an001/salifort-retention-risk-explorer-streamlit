# Docker Compose Validation Evidence

- `docker compose config` validates the local/dev Compose file.
- Services: `api`, `streamlit`, optional `mlflow` profile.
- Ports: `8000` for FastAPI, `8501` for Streamlit, `5000` for MLflow UI when the profile is enabled.
- Generated model artifacts are mounted from local `mlops/`; they are not baked into images.
- Boundary: Local/dev MLOps Mini-Lab evidence only; not production HR and not an employment decision system.
