"""Export the local/dev MLOps lab champion for hosted Streamlit inference."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL_PATH = REPO_ROOT / "mlops" / "models" / "champion_model.joblib"
SOURCE_EVALUATION_PATH = REPO_ROOT / "mlops" / "reports" / "evaluation_summary.json"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "mlops_lab_online"
OUTPUT_MODEL_PATH = OUTPUT_DIR / "champion_model.joblib"
OUTPUT_METADATA_PATH = OUTPUT_DIR / "model_metadata.json"

REQUIRED_INPUT_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    "promotion_last_5years",
    "department",
    "salary",
]
NORMALIZED_FEATURE_COLUMNS = [
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    "promotion_last_5years",
    "department",
    "salary_level",
    "overworked",
    "project_intensity",
    "career_stall_flag",
    "undervalued_flag",
    "tenure_x_projects",
]
PUBLIC_APP_BOUNDARY = (
    "The public Streamlit app remains artifact-backed with weighted XGBoost at threshold 0.29. "
    "This packaged artifact is only for the hosted MLOps Lab online demo."
)
RESPONSIBLE_USE_NOTE = (
    "Portfolio demonstration and human review support only. This artifact is not an employment decision system."
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python scripts/mlops_run_pipeline.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_from_champion(champion: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("best_"): value
        for key, value in champion.items()
        if key.startswith("best_") and key != "best_threshold"
    }


def export_streamlit_model_artifact() -> dict[str, Any]:
    if not SOURCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_MODEL_PATH}. Run python scripts/mlops_run_pipeline.py first."
        )
    summary = _read_json(SOURCE_EVALUATION_PATH)
    champion = summary.get("lab_champion")
    if not champion:
        raise ValueError(
            "Evaluation summary does not include lab_champion. "
            "Run python scripts/mlops_run_pipeline.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_MODEL_PATH, OUTPUT_MODEL_PATH)
    metadata = {
        "model_name": champion.get("model_name", "unknown_lab_champion"),
        "model_version": "mlops_lab_online_v1",
        "model_scope": "mlops-lab-online-demo",
        "source": "local/dev MLOps Mini-Lab champion export",
        "artifact_path": "artifacts/mlops_lab_online/champion_model.joblib",
        "threshold": champion.get("best_threshold", 0.6),
        "metrics": _metrics_from_champion(champion),
        "required_input_columns": REQUIRED_INPUT_COLUMNS,
        "normalized_feature_columns": NORMALIZED_FEATURE_COLUMNS,
        "mode": champion.get("mode", "operational"),
        "public_app_boundary": PUBLIC_APP_BOUNDARY,
        "responsible_use_note": RESPONSIBLE_USE_NOTE,
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    OUTPUT_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    metadata = export_streamlit_model_artifact()
    size = OUTPUT_MODEL_PATH.stat().st_size
    print("Exported Streamlit packaged MLOps Lab model")
    print(f"model: artifacts/mlops_lab_online/champion_model.joblib ({size} bytes)")
    print("metadata: artifacts/mlops_lab_online/model_metadata.json")
    print(f"model_name: {metadata['model_name']}")
    print(f"threshold: {metadata['threshold']}")
    print(PUBLIC_APP_BOUNDARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
