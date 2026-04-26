# GitHub Actions Evidence

The `CI` workflow includes:

- `app-runtime-checks`: installs Streamlit app requirements and compiles app runtime files.
- `mlops-tests`: installs app and MLOps dependencies, compiles MLOps/API/Airflow files, validates the DAG, and runs contract tests.
- `docker-config-check`: validates default Docker Compose config and the optional MLflow profile config.

The workflow does not deploy, publish Docker images, upload generated model artifacts, install Airflow, or run production HR workflows.

Boundary: Local/dev MLOps Mini-Lab evidence only; not production HR and not an employment decision system.
