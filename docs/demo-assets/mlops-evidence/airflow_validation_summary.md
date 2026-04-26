# Airflow DAG Validation Evidence

- DAG ID: `salifort_mlops_mini_lab_pipeline`
- Task order: `prepare_data >> train_model >> evaluate_model >> validate_api_contract`
- Static validator: `python scripts/validate_mlops_airflow_dag.py`
- Airflow is not installed as a required dependency.
- Streamlit does not trigger this DAG.
- Boundary: Local/dev MLOps Mini-Lab evidence only; not production HR and not an employment decision system.
