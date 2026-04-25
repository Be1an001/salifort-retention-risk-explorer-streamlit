from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = REPO_ROOT / "orchestration" / "airflow" / "dags" / "salifort_mlops_pipeline.py"
DAG_ID = "salifort_mlops_mini_lab_pipeline"
EXPECTED_TASK_IDS = (
    "prepare_data",
    "train_model",
    "evaluate_model",
    "validate_api_contract",
)
FORBIDDEN_SNIPPETS = (
    "deploy_to_production",
    "send_hr_alert",
    "artifacts/v2/metadata.json write",
    "streamlit run",
    "docker compose up",
    "git push",
    "employment decision",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_static_contract() -> dict[str, object]:
    _require(DAG_PATH.exists(), f"DAG file not found: {DAG_PATH}")
    py_compile.compile(str(DAG_PATH), doraise=True)
    text = DAG_PATH.read_text(encoding="utf-8")
    lower_text = text.lower()

    _require(DAG_ID in text, f"Expected DAG ID missing: {DAG_ID}")
    missing_tasks = [task_id for task_id in EXPECTED_TASK_IDS if task_id not in text]
    _require(not missing_tasks, "Missing task IDs: " + ", ".join(missing_tasks))
    _require(
        "prepare_data >> train_model >> evaluate_model >> validate_api_contract" in text,
        "Expected dependency chain is not represented.",
    )

    forbidden_found = [
        snippet for snippet in FORBIDDEN_SNIPPETS if snippet.lower() in lower_text
    ]
    _require(
        not forbidden_found,
        "Forbidden snippets found in DAG: " + ", ".join(forbidden_found),
    )
    _require(
        "artifacts/v2" not in lower_text,
        "DAG should not target artifacts/v2.",
    )
    _require(
        "streamlit" not in lower_text,
        "DAG should not trigger Streamlit.",
    )
    return {
        "dag_path": str(DAG_PATH),
        "dag_id": DAG_ID,
        "task_ids": EXPECTED_TASK_IDS,
        "static_contract_ok": True,
    }


def maybe_validate_airflow_import() -> dict[str, object]:
    if importlib.util.find_spec("airflow") is None:
        return {
            "airflow_import_checked": False,
            "airflow_import_skipped": True,
            "message": "apache-airflow is not installed; skipped optional DAG import.",
        }

    spec = importlib.util.spec_from_file_location("salifort_mlops_pipeline", DAG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {DAG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["salifort_mlops_pipeline"] = module
    spec.loader.exec_module(module)
    dag_objects = getattr(module, "DAG_OBJECTS", [])
    task_ids = []
    if dag_objects:
        task_ids = sorted(task.task_id for task in dag_objects[0].tasks)
    return {
        "airflow_import_checked": True,
        "airflow_import_skipped": False,
        "dag_count": len(dag_objects),
        "task_ids": task_ids,
    }


def main() -> int:
    static_result = validate_static_contract()
    import_result = maybe_validate_airflow_import()
    print("Salifort MLOps Airflow DAG validation")
    print(f"dag file: {static_result['dag_path']}")
    print(f"dag id: {static_result['dag_id']}")
    print(f"task ids: {', '.join(static_result['task_ids'])}")
    if import_result["airflow_import_skipped"]:
        print(import_result["message"])
    else:
        print(f"airflow import task ids: {', '.join(import_result['task_ids'])}")
    print("static contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
