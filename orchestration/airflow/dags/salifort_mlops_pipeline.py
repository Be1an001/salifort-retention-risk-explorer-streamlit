from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

DAG_ID = "salifort_mlops_mini_lab_pipeline"
PROJECT_ROOT = os.getenv(
    "SALIFORT_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[3]),
)

try:  # pragma: no cover - Airflow is optional for this portfolio repo.
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ModuleNotFoundError:  # pragma: no cover - exercised by static validation.
    AIRFLOW_AVAILABLE = False
    DAG_OBJECTS = []
else:  # pragma: no cover - depends on optional local Airflow installation.
    AIRFLOW_AVAILABLE = True

    def _project_command(script_path: str) -> str:
        return f'cd "$SALIFORT_PROJECT_ROOT" && python {script_path}'

    with DAG(
        dag_id=DAG_ID,
        description=(
            "Local/dev orchestration for the Salifort MLOps Mini-Lab pipeline. "
            "This DAG runs lab CLI scripts only and does not update public app artifacts."
        ),
        schedule=None,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["salifort", "mlops", "portfolio-demo", "local-dev"],
    ) as dag:
        prepare_data = BashOperator(
            task_id="prepare_data",
            bash_command=_project_command("scripts/mlops_01_prepare_data.py"),
            env={"SALIFORT_PROJECT_ROOT": PROJECT_ROOT},
        )

        train_model = BashOperator(
            task_id="train_model",
            bash_command=_project_command("scripts/mlops_02_train_model.py"),
            env={"SALIFORT_PROJECT_ROOT": PROJECT_ROOT},
        )

        evaluate_model = BashOperator(
            task_id="evaluate_model",
            bash_command=_project_command("scripts/mlops_03_evaluate_model.py"),
            env={"SALIFORT_PROJECT_ROOT": PROJECT_ROOT},
        )

        validate_api_contract = BashOperator(
            task_id="validate_api_contract",
            bash_command=(
                'cd "$SALIFORT_PROJECT_ROOT" && python -m pytest '
                "tests/test_api_contract.py tests/test_prediction_service.py"
            ),
            env={"SALIFORT_PROJECT_ROOT": PROJECT_ROOT},
        )

        prepare_data >> train_model >> evaluate_model >> validate_api_contract

    DAG_OBJECTS = [dag]
