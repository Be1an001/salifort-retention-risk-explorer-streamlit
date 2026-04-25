"""Prediction placeholders for future Salifort MLOps lab models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_lab_model(model_path: str | Path | None = None) -> Any:
    """Placeholder for loading a future lab model artifact."""

    raise NotImplementedError(
        "Lab model loading is not implemented in Phase 1. "
        f"Requested model path: {model_path!s}."
    )


def predict_placeholder_not_implemented() -> None:
    """Make accidental Phase 1 prediction attempts fail loudly."""

    raise NotImplementedError(
        "Prediction serving is intentionally not implemented in Phase 1. "
        "Future phases should load lab models without touching artifacts/v2."
    )
