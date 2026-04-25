"""Training placeholders for future Salifort MLOps phases.

Phase 1 deliberately does not fit models. These helpers describe the intended
plan so later phases can add sklearn/xgboost/mlflow dependencies without making
the current Streamlit app depend on them.
"""

from __future__ import annotations

from salifort_mlops.config import DEFAULT_THRESHOLD, MODEL_MODES
from salifort_mlops.features import get_model_feature_columns


def _validate_mode(mode: str) -> str:
    if mode not in MODEL_MODES:
        raise ValueError(f"Unsupported model mode {mode!r}. Expected one of {MODEL_MODES}.")
    return mode


def build_preprocessing_plan(mode: str = "operational") -> dict[str, object]:
    """Describe the future preprocessing split without importing sklearn."""

    mode = _validate_mode(mode)
    feature_columns = get_model_feature_columns(mode)
    categorical_features = ["department"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    return {
        "mode": mode,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "linear_model_steps": [
            "one-hot encode categorical features",
            "median-impute numeric features",
            "standard-scale numeric features",
        ],
        "tree_model_steps": [
            "one-hot encode categorical features",
            "median-impute numeric features",
        ],
    }


def describe_candidate_models() -> list[dict[str, object]]:
    """Return the candidate model plan inherited from the source workflow."""

    return [
        {
            "name": "log_reg_balanced",
            "family": "logistic_regression",
            "imbalance_strategy": "class_weight=balanced",
        },
        {
            "name": "log_reg_smote",
            "family": "logistic_regression",
            "imbalance_strategy": "SMOTE on training folds",
        },
        {
            "name": "rf_balanced",
            "family": "random_forest",
            "imbalance_strategy": "balanced_subsample",
        },
        {
            "name": "xgb_weighted",
            "family": "xgboost",
            "imbalance_strategy": "scale_pos_weight",
            "public_reference_threshold": DEFAULT_THRESHOLD,
        },
    ]


def train_placeholder_not_implemented() -> None:
    """Make accidental Phase 1 training attempts fail loudly."""

    raise NotImplementedError(
        "Model training is intentionally not implemented in Phase 1. "
        "Add training in a later MLOps phase outside the Streamlit runtime."
    )
