from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - Airflow is optional in this app repo.
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
except ModuleNotFoundError:  # pragma: no cover - exercised by validation script.
    AIRFLOW_AVAILABLE = False
    DAG_OBJECTS = []
else:  # pragma: no cover - depends on optional Airflow installation.
    from datetime import datetime

    from app.services.navigator_orchestration import (
        get_task_definition,
        get_workflow_definition,
        list_workflows,
    )

    AIRFLOW_AVAILABLE = True
    DAG_OBJECTS = []

    def _task_operator(dag: DAG, task_id: str):
        task = get_task_definition(task_id)
        command_hint = str(task.get("command_hint", ""))
        operator_id = task_id.replace("-", "_")
        if command_hint.startswith("python "):
            return BashOperator(
                task_id=operator_id,
                bash_command=command_hint,
                dag=dag,
            )
        return EmptyOperator(
            task_id=operator_id,
            dag=dag,
        )

    for workflow in list_workflows():
        if workflow["scheduler_eligibility"] == "not_scheduled_reviewer_interactive":
            continue
        workflow_definition = get_workflow_definition(workflow["workflow_id"])
        dag = DAG(
            dag_id=f"pace_navigator_{workflow['workflow_id']}",
            description=workflow["notes"],
            schedule=None,
            start_date=datetime(2026, 1, 1),
            catchup=False,
            tags=[
                "pace-navigator",
                "governed",
                workflow["workflow_kind"],
                workflow["runtime_mode"],
            ],
        )
        operators = {
            task_id: _task_operator(dag, task_id)
            for task_id in workflow_definition["task_ids"]
        }
        for task_id in workflow_definition["task_ids"]:
            task = get_task_definition(task_id)
            for dependency_id in task["dependencies"]:
                if dependency_id in operators:
                    operators[dependency_id] >> operators[task_id]
        globals()[dag.dag_id] = dag
        DAG_OBJECTS.append(dag)
