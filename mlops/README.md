# Salifort MLOps Mini-Lab

This folder is reserved for the local/dev MLOps Mini-Lab extension.

It is separate from `artifacts/v2/` and does not replace the public Streamlit app story:

- public reference model: weighted XGBoost
- selected public threshold: `0.29`
- current app runtime: artifact-backed Streamlit pages

Future phases may write processed lab datasets, trained lab models, and report files here. Those outputs are for local development and portfolio demonstration only.

The existing Streamlit pages should not depend on this folder to open, and Streamlit should not trigger training, Docker, Airflow, MLflow, or FastAPI workflows.
