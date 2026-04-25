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


def main() -> int:
    report = run_prepare_data()
    print_summary(report)
    print("Later phases not implemented: training, MLflow, FastAPI, Docker, Airflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
