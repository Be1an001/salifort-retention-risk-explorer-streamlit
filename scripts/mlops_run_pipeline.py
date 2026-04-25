from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mlops_01_prepare_data import print_summary, run_prepare_data  # noqa: E402
from mlops_02_train_model import print_summary as print_training_summary  # noqa: E402
from mlops_02_train_model import run_training  # noqa: E402
from mlops_03_evaluate_model import print_summary as print_evaluation_summary  # noqa: E402
from mlops_03_evaluate_model import run_evaluation_summary  # noqa: E402


def main() -> int:
    prepare_report = run_prepare_data()
    print_summary(prepare_report)
    training_payload = run_training()
    print_training_summary(training_payload)
    evaluation_summary = run_evaluation_summary()
    print_evaluation_summary(evaluation_summary)
    print("Later phases not implemented: FastAPI, Docker, Airflow, Streamlit page registration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
