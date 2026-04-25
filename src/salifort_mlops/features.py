"""Feature engineering helpers for future Salifort lab modeling phases."""

from __future__ import annotations

import pandas as pd

from salifort_mlops.config import (
    MODEL_MODES,
    OPERATIONAL_EXCLUDED_FEATURES,
    SALARY_LEVEL_MAP,
    TARGET_COLUMN,
)
from salifort_mlops.data_prep import clean_salifort_data

BASE_FEATURE_COLUMNS = (
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
)

SURVEY_RICH_EXTRA_FEATURE_COLUMNS = (
    "satisfaction_level",
    "burnout_index",
    "effort_reward_gap",
    "low_satisfaction_high_eval",
)


def _validate_mode(mode: str) -> str:
    if mode not in MODEL_MODES:
        raise ValueError(f"Unsupported model mode {mode!r}. Expected one of {MODEL_MODES}.")
    return mode


def build_hr_features(df: pd.DataFrame, mode: str = "operational") -> pd.DataFrame:
    """Build deterministic HR features from cleaned or raw Salifort data.

    Operational mode intentionally excludes direct satisfaction-based features
    from the final model feature set, matching the deployment-like source
    workflow. Survey-rich mode keeps those explanatory features for comparison.
    """

    mode = _validate_mode(mode)
    data = clean_salifort_data(df)
    return add_engineered_features(data, mode=mode)


def add_engineered_features(df: pd.DataFrame, mode: str = "operational") -> pd.DataFrame:
    """Add deterministic lab features to a standardized Salifort frame.

    This helper assumes column names are already normalized. It does not drop
    duplicates, which keeps inference batches aligned with request rows.
    """

    mode = _validate_mode(mode)
    data = df.copy()
    salary_text = data["salary"].astype(str)
    data["salary_level"] = salary_text.map(SALARY_LEVEL_MAP)
    data["overworked"] = (data["average_monthly_hours"] > 175).astype(int)
    project_count = data["number_project"].where(data["number_project"] != 0)
    data["project_intensity"] = (data["average_monthly_hours"] / project_count).fillna(0)
    data["career_stall_flag"] = (
        (data["tenure"] >= 4) & (data["promotion_last_5years"] == 0)
    ).astype(int)
    data["undervalued_flag"] = (
        (data["number_project"] >= 5)
        & (data["last_evaluation"] >= 0.80)
        & (data["promotion_last_5years"] == 0)
    ).astype(int)
    data["tenure_x_projects"] = data["tenure"] * data["number_project"]

    if mode == "survey_rich":
        median_hours = data["average_monthly_hours"].median()
        data["burnout_index"] = (
            data["average_monthly_hours"] / median_hours
        ) * (1 - data["satisfaction_level"])
        data["effort_reward_gap"] = data["last_evaluation"] - data["satisfaction_level"]
        data["low_satisfaction_high_eval"] = (
            (data["satisfaction_level"] < 0.45) & (data["last_evaluation"] > 0.75)
        ).astype(int)

    return data


def get_model_feature_columns(mode: str = "operational") -> list[str]:
    """Return model feature columns for the requested future training mode."""

    mode = _validate_mode(mode)
    columns = list(BASE_FEATURE_COLUMNS)
    if mode == "survey_rich":
        columns.extend(SURVEY_RICH_EXTRA_FEATURE_COLUMNS)
    return [column for column in columns if column not in OPERATIONAL_EXCLUDED_FEATURES or mode != "operational"]


def split_features_target(
    df: pd.DataFrame,
    mode: str = "operational",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target vector for later training phases."""

    feature_df = build_hr_features(df, mode=mode)
    feature_columns = get_model_feature_columns(mode=mode)
    missing = [column for column in feature_columns if column not in feature_df.columns]
    if missing:
        raise ValueError("Feature engineering did not create expected columns: " + ", ".join(missing))
    return feature_df[feature_columns].copy(), feature_df[TARGET_COLUMN].copy()
