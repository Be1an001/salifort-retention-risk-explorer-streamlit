from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = REPO_ROOT / "orchestration" / "airflow" / "dags" / "salifort_mlops_pipeline.py"


def _dag_text() -> str:
    assert DAG_PATH.exists()
    return DAG_PATH.read_text(encoding="utf-8")


def test_mlops_airflow_dag_file_exists_and_declares_dag_id() -> None:
    text = _dag_text()
    assert "salifort_mlops_mini_lab_pipeline" in text
    assert "schedule=None" in text
    assert "catchup=False" in text


def test_mlops_airflow_dag_declares_expected_tasks() -> None:
    text = _dag_text()
    for task_id in (
        "prepare_data",
        "train_model",
        "evaluate_model",
        "validate_api_contract",
    ):
        assert f'task_id="{task_id}"' in text


def test_mlops_airflow_dag_dependency_chain_is_represented() -> None:
    text = _dag_text()
    assert "prepare_data >> train_model >> evaluate_model >> validate_api_contract" in text


def test_mlops_airflow_dag_calls_existing_cli_scripts_only() -> None:
    text = _dag_text()
    assert "scripts/mlops_01_prepare_data.py" in text
    assert "scripts/mlops_02_train_model.py" in text
    assert "scripts/mlops_03_evaluate_model.py" in text
    assert "tests/test_api_contract.py tests/test_prediction_service.py" in text


def test_mlops_airflow_dag_avoids_forbidden_runtime_actions() -> None:
    lower_text = _dag_text().lower()
    forbidden = (
        "deploy_to_production",
        "send_hr_alert",
        "artifacts/v2/metadata.json write",
        "streamlit run",
        "docker compose up",
        "git push",
        "employment decision",
    )
    assert not [snippet for snippet in forbidden if snippet in lower_text]
    assert "artifacts/v2" not in lower_text
    assert "streamlit" not in lower_text
