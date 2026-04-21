from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DAG_MODULE_PATH = (
    REPO_ROOT
    / "orchestration"
    / "airflow"
    / "dags"
    / "pace_navigator_governed_workflows.py"
)


def main() -> int:
    try:
        spec = importlib.util.spec_from_file_location(
            "pace_navigator_governed_workflows",
            DAG_MODULE_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load DAG scaffold from {DAG_MODULE_PATH}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"Airflow scaffold validation failed: {type(exc).__name__}: {exc}")
        return 1

    airflow_available = bool(getattr(module, "AIRFLOW_AVAILABLE", False))
    dag_objects = getattr(module, "DAG_OBJECTS", [])
    print("Airflow scaffold validation passed.")
    print(f"- Airflow installed: {airflow_available}")
    print(f"- DAG objects created: {len(dag_objects)}")
    if not airflow_available:
        print("- Airflow is optional; scaffold imported safely without DAG object creation.")
    else:
        print("- DAG ids: " + ", ".join(dag.dag_id for dag in dag_objects))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
