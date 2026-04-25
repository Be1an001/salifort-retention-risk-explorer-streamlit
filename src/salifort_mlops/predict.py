"""Prediction placeholders for future Salifort MLOps lab models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from salifort_mlops.config import LAB_MODELS_DIR


def load_lab_model(model_path: str | Path | None = None) -> Any:
    """Load a future lab model artifact from ``mlops/models``.

    This helper deliberately does not read or modify ``artifacts/v2``.
    """

    resolved_path = Path(model_path) if model_path is not None else LAB_MODELS_DIR / "lab_champion.joblib"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Lab model artifact not found at {resolved_path}. "
            "Run scripts/mlops_02_train_model.py first."
        )
    return joblib.load(resolved_path)


def predict_placeholder_not_implemented() -> None:
    """Make accidental Phase 1 prediction attempts fail loudly."""

    raise NotImplementedError(
        "Prediction serving is intentionally not implemented in Phase 1. "
        "Future phases should load lab models without touching artifacts/v2."
    )
