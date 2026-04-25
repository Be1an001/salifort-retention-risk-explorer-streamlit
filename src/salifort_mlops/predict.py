"""Prediction helpers for optional Salifort MLOps lab serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd

from salifort_mlops.config import (
    DEFAULT_THRESHOLD,
    LEGACY_LAB_CHAMPION_MODEL_PATH,
    STABLE_CHAMPION_MODEL_PATH,
    TARGET_COLUMN,
)
from salifort_mlops.data_prep import standardize_columns
from salifort_mlops.features import add_engineered_features, get_model_feature_columns
from salifort_mlops.schemas import validate_category_values


def load_lab_model(model_path: str | Path | None = None) -> Any:
    """Load a lab model artifact without touching public Streamlit artifacts.

    The stable serving path is ``mlops/models/champion_model.joblib``. The
    Phase 3 ``lab_champion.joblib`` path is retained as a fallback for local
    workspaces created before the stable alias existed.
    """

    if model_path is not None:
        resolved_path = Path(model_path)
    elif STABLE_CHAMPION_MODEL_PATH.exists():
        resolved_path = STABLE_CHAMPION_MODEL_PATH
    else:
        resolved_path = LEGACY_LAB_CHAMPION_MODEL_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Lab model artifact not found at {resolved_path}. "
            "Run python scripts/mlops_run_pipeline.py first."
        )
    return joblib.load(resolved_path)


def prepare_inference_frame(
    records: Iterable[dict[str, Any]],
    mode: str = "operational",
) -> pd.DataFrame:
    """Normalize request records and return model-ready feature columns."""

    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("At least one record is required for prediction.")

    for row in rows:
        row.setdefault(TARGET_COLUMN, 0)
    data = pd.DataFrame(rows)
    normalized = standardize_columns(data)
    category_result = validate_category_values(normalized)
    category_result.raise_for_errors()
    engineered = add_engineered_features(normalized, mode=mode)
    feature_columns = get_model_feature_columns(mode=mode)
    missing = [column for column in feature_columns if column not in engineered.columns]
    if missing:
        raise ValueError("Inference records are missing required features: " + ", ".join(missing))
    return engineered[feature_columns].copy()


def assign_review_band(probability: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map an attrition probability into a simple human-review band."""

    if probability >= threshold:
        return "high"
    if probability >= threshold * 0.75:
        return "medium"
    return "low"


def predict_proba(
    model: Any,
    records: Iterable[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    mode: str = "operational",
) -> list[dict[str, Any]]:
    """Return probability, flag, and review band for request records."""

    features = prepare_inference_frame(records, mode=mode)
    probabilities = model.predict_proba(features)[:, 1]
    return [
        {
            "attrition_probability": float(probability),
            "threshold": float(threshold),
            "high_risk_flag": bool(probability >= threshold),
            "review_band": assign_review_band(float(probability), threshold),
        }
        for probability in probabilities
    ]


def predict_placeholder_not_implemented() -> None:
    """Retained for callers that explicitly request a placeholder."""

    raise NotImplementedError(
        "Use the optional FastAPI service or predict_proba helper for lab serving. "
        "Prediction must remain separate from the Streamlit artifact runtime."
    )
