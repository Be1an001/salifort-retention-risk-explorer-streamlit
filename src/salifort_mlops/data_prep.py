"""Data preparation helpers for the Salifort MLOps Mini-Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from salifort_mlops.config import (
    LAB_MODELS_DIR,
    LAB_REPORTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_CANDIDATE_PATHS,
    TARGET_COLUMN,
)
from salifort_mlops.schemas import (
    normalize_column_names,
    validate_category_values,
    validate_columns,
)


def ensure_directories() -> None:
    """Create local lab output directories for later phases."""

    for path in (PROCESSED_DATA_DIR, LAB_MODELS_DIR, LAB_REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _resolve_raw_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Raw data file was not found: {resolved}")
        return resolved

    for candidate in RAW_DATA_CANDIDATE_PATHS:
        if candidate.exists():
            return candidate

    candidates = ", ".join(str(candidate) for candidate in RAW_DATA_CANDIDATE_PATHS)
    raise FileNotFoundError(f"No Salifort raw data file found. Checked: {candidates}")


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Salifort CSV from an explicit path or known repo candidates."""

    return pd.read_csv(_resolve_raw_data_path(path))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy source column names without mutating the input frame."""

    result = validate_columns(df, allow_raw_aliases=True)
    result.raise_for_errors()
    return normalize_column_names(df)


def clean_salifort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, validate categories, drop duplicates, and reset index."""

    cleaned = standardize_columns(df)
    category_result = validate_category_values(cleaned)
    category_result.raise_for_errors()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True).copy()
    return cleaned


def build_data_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Return a lightweight profile suitable for smoke checks and reports."""

    profile: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "duplicate_count": int(df.duplicated().sum()),
        "missing_values": {
            column: int(count) for column, count in df.isna().sum().to_dict().items()
        },
    }
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
        profile["target_counts"] = {
            str(key): int(value) for key, value in counts.to_dict().items()
        }
        profile["target_rate"] = float(df[TARGET_COLUMN].mean())
    if "department" in df.columns:
        profile["department_values"] = sorted(df["department"].dropna().astype(str).unique())
    if "salary" in df.columns:
        profile["salary_values"] = sorted(df["salary"].dropna().astype(str).unique())
    return profile


def split_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a deterministic stratified train/test split using pandas only."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Cannot split data because target column {TARGET_COLUMN!r} is missing.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    class_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    if len(class_counts) < 2:
        raise ValueError("Stratified split requires at least two target classes.")
    if (class_counts < 2).any():
        raise ValueError("Stratified split requires at least two rows per target class.")

    test_parts = []
    for _, group in df.groupby(TARGET_COLUMN, sort=False):
        test_count = max(1, round(len(group) * test_size))
        if test_count >= len(group):
            raise ValueError(
                "test_size leaves no training rows for at least one target class."
            )
        test_parts.append(group.sample(n=test_count, random_state=random_state))
    test_df = pd.concat(test_parts).sort_index()
    train_df = df.drop(index=test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
