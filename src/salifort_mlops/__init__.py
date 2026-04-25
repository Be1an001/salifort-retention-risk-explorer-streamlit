"""Lightweight MLOps package foundation for the Salifort Mini-Lab.

This package is intentionally independent from the Streamlit runtime. It
contains reusable data, feature, schema, evaluation, training, and prediction
helpers for future local/dev MLOps phases.
"""

from salifort_mlops.config import (
    DEFAULT_THRESHOLD,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    RANDOM_SEED,
    TARGET_COLUMN,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "FALSE_NEGATIVE_COST",
    "FALSE_POSITIVE_COST",
    "RANDOM_SEED",
    "TARGET_COLUMN",
]
