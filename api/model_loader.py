"""Lazy model loading for the optional Salifort MLOps Mini-Lab API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from api.schemas import EmployeeFeatures
from salifort_mlops.config import (
    DEFAULT_THRESHOLD,
    LAB_MODELS_DIR,
    LAB_REPORTS_DIR,
    LEGACY_LAB_CHAMPION_MODEL_PATH,
    STABLE_CHAMPION_MODEL_PATH,
)
from salifort_mlops.predict import load_lab_model, predict_proba

SERVICE_NAME = "salifort-mlops-mini-lab-api"
MODEL_SCOPE = "mlops-mini-lab"
MISSING_MODEL_MESSAGE = (
    "Lab champion model artifact is not available. Run python scripts/mlops_run_pipeline.py "
    "to prepare data, train candidates, and export the local/dev lab model."
)

EVALUATION_SUMMARY_PATH = LAB_REPORTS_DIR / "evaluation_summary.json"
TRAINING_RESULTS_JSON_PATH = LAB_REPORTS_DIR / "training_results.json"

_MODEL_CACHE: Any | None = None
_MODEL_PATH_CACHE: Path | None = None


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before a lab model is exported."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _lab_champion_metadata() -> dict[str, Any]:
    env_metadata_path = _env_path("SALIFORT_MODEL_METADATA_PATH")
    if env_metadata_path:
        metadata = _read_json(env_metadata_path)
        if metadata.get("lab_champion"):
            return metadata["lab_champion"]
        if metadata:
            return metadata
    evaluation_summary = _read_json(EVALUATION_SUMMARY_PATH)
    if evaluation_summary.get("lab_champion"):
        return evaluation_summary["lab_champion"]
    training_summary = _read_json(TRAINING_RESULTS_JSON_PATH)
    if training_summary.get("lab_champion"):
        return training_summary["lab_champion"]
    return {}


def _candidate_model_paths() -> list[Path]:
    champion = _lab_champion_metadata()
    model_name = champion.get("model_name")
    paths = []
    env_model_path = _env_path("SALIFORT_MODEL_PATH")
    if env_model_path:
        paths.append(env_model_path)
    paths.extend([STABLE_CHAMPION_MODEL_PATH, LEGACY_LAB_CHAMPION_MODEL_PATH])
    if model_name:
        paths.append(LAB_MODELS_DIR / f"{model_name}.joblib")
    return paths


def _resolve_model_path() -> Path | None:
    for path in _candidate_model_paths():
        if path.exists():
            return path
    return None


def get_model_state() -> dict[str, Any]:
    """Return current lazy-loading state without failing when files are absent."""

    path = _resolve_model_path()
    return {
        "model_available": path is not None,
        "model_loaded": _MODEL_CACHE is not None,
        "model_path": str(path) if path else None,
    }


def load_model() -> Any:
    """Load and cache the lab champion model on first use."""

    global _MODEL_CACHE, _MODEL_PATH_CACHE
    path = _resolve_model_path()
    if path is None:
        raise ModelUnavailableError(MISSING_MODEL_MESSAGE)
    if _MODEL_CACHE is None or _MODEL_PATH_CACHE != path:
        _MODEL_CACHE = load_lab_model(path)
        _MODEL_PATH_CACHE = path
    return _MODEL_CACHE


def is_model_available() -> bool:
    """Return whether a lab champion model artifact is present."""

    return _resolve_model_path() is not None


def get_model_info() -> dict[str, Any]:
    """Return model metadata and serving notes for API responses."""

    champion = _lab_champion_metadata()
    path = _resolve_model_path()
    threshold = champion.get("best_threshold", DEFAULT_THRESHOLD)
    metrics = {
        key.removeprefix("best_"): value
        for key, value in champion.items()
        if key.startswith("best_") and key != "best_threshold"
    }
    return {
        "service": SERVICE_NAME,
        "model_available": path is not None,
        "model_name": champion.get("model_name") if champion else None,
        "model_version": path.stem if path else None,
        "threshold": float(threshold) if threshold is not None else None,
        "metrics": metrics,
        "artifact_paths": {
            "model": str(path) if path else None,
            "evaluation_summary": str(EVALUATION_SUMMARY_PATH)
            if EVALUATION_SUMMARY_PATH.exists()
            else None,
            "training_results": str(TRAINING_RESULTS_JSON_PATH)
            if TRAINING_RESULTS_JSON_PATH.exists()
            else None,
        },
        "message": "Lab model is available." if path else MISSING_MODEL_MESSAGE,
    }


def _payload_to_record(payload: EmployeeFeatures | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, EmployeeFeatures):
        return payload.model_dump()
    return EmployeeFeatures.model_validate(payload).model_dump()


def predict_one(payload: EmployeeFeatures | dict[str, Any]) -> dict[str, Any]:
    """Predict one normalized or alias-compatible payload."""

    model = load_model()
    info = get_model_info()
    threshold = float(info["threshold"] or DEFAULT_THRESHOLD)
    record = _payload_to_record(payload)
    prediction = predict_proba(model, [record], threshold=threshold)[0]
    prediction.update(
        {
            "model_name": info["model_name"] or "unknown_lab_model",
            "model_version": info["model_version"] or "unknown",
        }
    )
    return prediction


def predict_batch(payloads: list[EmployeeFeatures | dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict a list of normalized or alias-compatible payloads."""

    model = load_model()
    info = get_model_info()
    threshold = float(info["threshold"] or DEFAULT_THRESHOLD)
    records = [_payload_to_record(payload) for payload in payloads]
    predictions = predict_proba(model, records, threshold=threshold)
    for prediction in predictions:
        prediction.update(
            {
                "model_name": info["model_name"] or "unknown_lab_model",
                "model_version": info["model_version"] or "unknown",
            }
        )
    return predictions
