"""Export sanitized, lightweight evidence for the MLOps Mini-Lab demo.

This script reads local/dev generated reports when present and writes a
committable evidence snapshot under docs/demo-assets/mlops-evidence/.
It never copies model binaries, mlruns, secrets, uploaded CSVs, or large data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "mlops" / "reports"
MODELS_DIR = PROJECT_ROOT / "mlops" / "models"
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "demo-assets" / "mlops-evidence"

PUBLIC_APP_TRUTH = "Public app truth remains weighted XGBoost at threshold 0.29."
BOUNDARY = "Local/dev MLOps Mini-Lab evidence only; not production HR and not an employment decision system."
PLACEHOLDER = (
    "Evidence snapshot not generated yet. Run python scripts/mlops_run_pipeline.py, "
    "then python scripts/export_mlops_evidence_pack.py."
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _base_metadata() -> dict[str, str]:
    return {
        "evidence_generated_at": _now(),
        "source": "local/dev MLOps Mini-Lab run",
        "boundary": BOUNDARY,
        "public_app_truth": PUBLIC_APP_TRUTH,
    }


def _repo_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        parts = [part for part in candidate.parts if part not in {candidate.anchor}]
        for anchor in ("mlops", "docs", "api", "scripts", "data"):
            if anchor in parts:
                return Path(*parts[parts.index(anchor) :]).as_posix()
    return None


def _write_json(name: str, payload: dict[str, Any]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(name: str, content: str) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")


def _placeholder_payload(kind: str) -> dict[str, Any]:
    return {
        **_base_metadata(),
        "status": "placeholder",
        "evidence_type": kind,
        "message": PLACEHOLDER,
    }


def export_pipeline_summary() -> None:
    profile = _read_json(REPORTS_DIR / "data_profile.json")
    if not profile:
        _write_json("pipeline_run_summary.json", _placeholder_payload("pipeline_run_summary"))
        return

    raw = profile.get("raw", {})
    clean = profile.get("clean", {})
    split = profile.get("split", {})
    payload = {
        **_base_metadata(),
        "status": "generated",
        "pipeline": profile.get("pipeline", "mlops_01_prepare_data"),
        "mode": profile.get("mode", "operational"),
        "raw_rows": raw.get("row_count"),
        "clean_rows": clean.get("row_count"),
        "duplicates_removed": profile.get("duplicates_removed", raw.get("duplicate_count")),
        "train_rows": split.get("train_rows"),
        "test_rows": split.get("test_rows"),
        "target_counts": clean.get("target_counts") or raw.get("target_counts"),
        "generated_artifacts": [
            "mlops/data/processed/train.parquet",
            "mlops/data/processed/test.parquet",
            "mlops/reports/data_profile.json",
            "mlops/reports/training_results.json",
            "mlops/reports/evaluation_summary.json",
            "mlops/reports/model_card.md",
            "mlops/models/champion_model.joblib",
        ],
        "note": "Generated outputs remain local and gitignored; this file is a sanitized evidence snapshot.",
    }
    _write_json("pipeline_run_summary.json", payload)


def export_training_summary() -> None:
    evaluation = _read_json(REPORTS_DIR / "evaluation_summary.json")
    training = _read_json(REPORTS_DIR / "training_results.json")
    mlflow = _read_json(REPORTS_DIR / "mlflow_summary.json")
    champion = evaluation.get("lab_champion") or training.get("lab_champion") or {}
    if not champion:
        _write_json("training_evaluation_summary.json", _placeholder_payload("training_evaluation_summary"))
        return

    metrics = {
        "accuracy": champion.get("best_accuracy"),
        "precision": champion.get("best_precision"),
        "recall": champion.get("best_recall"),
        "f1": champion.get("best_f1"),
        "f2": champion.get("best_f2"),
        "roc_auc": champion.get("best_roc_auc"),
        "pr_auc": champion.get("best_pr_auc"),
        "tn": champion.get("best_tn"),
        "fp": champion.get("best_fp"),
        "fn": champion.get("best_fn"),
        "tp": champion.get("best_tp"),
        "cost": champion.get("best_cost"),
    }
    payload = {
        **_base_metadata(),
        "status": "generated",
        "candidate_models": training.get("candidate_models", []),
        "lab_champion": champion.get("model_name"),
        "lab_threshold": champion.get("best_threshold"),
        "metrics": metrics,
        "mlflow_tracking_evidence": {
            "experiment_name": mlflow.get("experiment_name", "salifort-mlops-mini-lab"),
            "logged_candidate_count": mlflow.get("logged_candidate_count", len(training.get("candidate_models", []))),
            "tracking_uri": "local mlruns/ directory (gitignored)",
            "runs_are_committed": False,
        },
        "public_reference_note": PUBLIC_APP_TRUTH,
        "responsible_use_note": BOUNDARY,
    }
    _write_json("training_evaluation_summary.json", payload)


def export_fastapi_examples() -> None:
    evaluation = _read_json(REPORTS_DIR / "evaluation_summary.json")
    training = _read_json(REPORTS_DIR / "training_results.json")
    champion = evaluation.get("lab_champion") or training.get("lab_champion") or {}
    model_path = MODELS_DIR / "champion_model.joblib"
    model_available = model_path.exists()
    metadata_available = bool(champion)

    if model_available:
        message = "Service is healthy. Model artifacts are available and the model will load on first prediction."
    else:
        message = "Service is healthy, but lab model artifacts are missing. Run python scripts/mlops_run_pipeline.py."
    _write_json(
        "fastapi_health_example.json",
        {
            **_base_metadata(),
            "status": "ok",
            "model_artifact_available": model_available,
            "model_metadata_available": metadata_available,
            "model_loaded_in_memory": False,
            "model_ready": model_available,
            "message": message,
        },
    )

    metrics = {
        key.removeprefix("best_"): value
        for key, value in champion.items()
        if key.startswith("best_") and key != "best_threshold"
    }
    _write_json(
        "fastapi_model_info_example.json",
        {
            **_base_metadata(),
            "service": "salifort-mlops-mini-lab-api",
            "model_available": model_available,
            "model_name": champion.get("model_name"),
            "model_version": "champion_model" if model_available else None,
            "threshold": champion.get("best_threshold"),
            "metrics": metrics,
            "artifact_paths": {
                "model": "mlops/models/champion_model.joblib" if model_available else None,
                "evaluation_summary": "mlops/reports/evaluation_summary.json" if evaluation else None,
                "training_results": "mlops/reports/training_results.json" if training else None,
            },
            "public_reference_note": PUBLIC_APP_TRUTH,
            "responsible_use_note": BOUNDARY,
        },
    )


def export_markdown_summaries() -> None:
    compose_text = _read_text(PROJECT_ROOT / "docker-compose.yml")
    has_mlflow = "mlflow:" in compose_text and "profiles:" in compose_text
    _write_text(
        "docker_compose_validation.md",
        f"""
# Docker Compose Validation Evidence

- `docker compose config` validates the local/dev Compose file.
- Services: `api`, `streamlit`{', optional `mlflow` profile' if has_mlflow else ''}.
- Ports: `8000` for FastAPI, `8501` for Streamlit{', `5000` for MLflow UI when the profile is enabled' if has_mlflow else ''}.
- Generated model artifacts are mounted from local `mlops/`; they are not baked into images.
- Boundary: {BOUNDARY}
""",
    )
    _write_text(
        "airflow_validation_summary.md",
        f"""
# Airflow DAG Validation Evidence

- DAG ID: `salifort_mlops_mini_lab_pipeline`
- Task order: `prepare_data >> train_model >> evaluate_model >> validate_api_contract`
- Static validator: `python scripts/validate_mlops_airflow_dag.py`
- Airflow is not installed as a required dependency.
- Streamlit does not trigger this DAG.
- Boundary: {BOUNDARY}
""",
    )
    _write_text(
        "github_actions_summary.md",
        f"""
# GitHub Actions Evidence

The `CI` workflow includes:

- `app-runtime-checks`: installs Streamlit app requirements and compiles app runtime files.
- `mlops-tests`: installs app and MLOps dependencies, compiles MLOps/API/Airflow files, validates the DAG, and runs contract tests.
- `docker-config-check`: validates default Docker Compose config and the optional MLflow profile config.

The workflow does not deploy, publish Docker images, upload generated model artifacts, install Airflow, or run production HR workflows.

Boundary: {BOUNDARY}
""",
    )


def export_readme() -> None:
    _write_text(
        "README.md",
        f"""
# MLOps Evidence Pack

This folder contains sanitized, lightweight evidence snapshots for the Salifort MLOps Mini-Lab.

It is intended for online reviewers who cannot run the local/dev pipeline on the user's computer.

Included evidence:

- `pipeline_run_summary.json`: data-prep and split summary.
- `training_evaluation_summary.json`: candidate model, lab champion, metric, and MLflow tracking summary.
- `fastapi_health_example.json`: sanitized `/health` response shape.
- `fastapi_model_info_example.json`: sanitized `/model-info` response shape with repo-relative artifact paths.
- `docker_compose_validation.md`: local/dev Docker Compose validation summary.
- `airflow_validation_summary.md`: local/dev DAG validation summary.
- `github_actions_summary.md`: CI validation summary.

Excluded by design:

- joblib model files
- `mlruns/`
- uploaded CSV files
- API keys, `.env`, or `.streamlit/secrets.toml`
- large generated data files
- local absolute paths

Boundary: {BOUNDARY}

Public app truth: {PUBLIC_APP_TRUTH}

Regenerate locally after running the MLOps pipeline:

```bash
python scripts/mlops_run_pipeline.py
python scripts/export_mlops_evidence_pack.py
```
""",
    )


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    export_pipeline_summary()
    export_training_summary()
    export_fastapi_examples()
    export_markdown_summaries()
    export_readme()
    print(f"Wrote sanitized MLOps evidence pack to {_repo_path(EVIDENCE_DIR)}")


if __name__ == "__main__":
    main()
