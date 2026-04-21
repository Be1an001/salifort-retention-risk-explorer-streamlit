# Scripts Folder

This folder contains offline builders and validation scripts.

These scripts are for local development and project review. The Streamlit app should not run model training, artifact building, Airflow tasks, or agent workflows during a visitor session.

## Common Script Groups

- `build_v2_artifacts.py`: builds generated model and explanation artifacts for `artifacts/v2/` when the modeling dependencies are available.
- `build_retrieval_pack.py` and `build_retrieval_index.py`: prepare and index the advanced Navigator retrieval corpus.
- `validate_*.py`: checks navigator registries, retrieval preparation, answer assembly, orchestration contracts, agent shell rules, and final readiness.

Some scripts may require local environment variables or optional modeling/API dependencies. Do not commit API keys, `.env` files, or generated secrets.
