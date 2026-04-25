"""Schema and lightweight data-contract checks for Salifort lab data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from salifort_mlops.config import (
    ALLOWED_DEPARTMENT_VALUES,
    ALLOWED_SALARY_VALUES,
    RAW_TO_NORMALIZED_COLUMNS,
    TARGET_COLUMN,
)

REQUIRED_RAW_COLUMNS = (
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    TARGET_COLUMN,
    "promotion_last_5years",
    "Department",
    "salary",
)

REQUIRED_NORMALIZED_COLUMNS = (
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    TARGET_COLUMN,
    "promotion_last_5years",
    "department",
    "salary",
)


@dataclass(frozen=True)
class ValidationResult:
    """Structured result for non-mutating data validation."""

    ok: bool
    missing_columns: tuple[str, ...] = ()
    unexpected_columns: tuple[str, ...] = ()
    invalid_categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        """Raise a clear exception when blocking validation failures exist."""

        messages: list[str] = []
        if self.missing_columns:
            messages.append("missing columns: " + ", ".join(self.missing_columns))
        if self.invalid_categories:
            details = [
                f"{column}={list(values)}"
                for column, values in self.invalid_categories.items()
            ]
            messages.append("invalid categories: " + "; ".join(details))
        if messages:
            raise ValueError("Salifort data validation failed: " + " | ".join(messages))


def get_required_raw_columns() -> tuple[str, ...]:
    """Return the original raw columns expected from the source dataset."""

    return REQUIRED_RAW_COLUMNS


def get_required_normalized_columns() -> tuple[str, ...]:
    """Return the normalized columns used by the lab package."""

    return REQUIRED_NORMALIZED_COLUMNS


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with legacy Salifort column names normalized."""

    return df.rename(columns=RAW_TO_NORMALIZED_COLUMNS).copy()


def _missing_columns(
    actual_columns: Iterable[str],
    *,
    allow_raw_aliases: bool,
) -> tuple[str, ...]:
    actual = set(actual_columns)
    missing: list[str] = []
    for column in REQUIRED_NORMALIZED_COLUMNS:
        if column in actual:
            continue
        raw_aliases = [
            raw for raw, normalized in RAW_TO_NORMALIZED_COLUMNS.items() if normalized == column
        ]
        if allow_raw_aliases and any(alias in actual for alias in raw_aliases):
            continue
        missing.append(column)
    return tuple(missing)


def validate_columns(
    df: pd.DataFrame,
    *,
    allow_raw_aliases: bool = True,
) -> ValidationResult:
    """Validate required columns while optionally accepting raw source aliases."""

    missing = _missing_columns(df.columns, allow_raw_aliases=allow_raw_aliases)
    normalized_or_raw = set(REQUIRED_NORMALIZED_COLUMNS)
    if allow_raw_aliases:
        normalized_or_raw.update(REQUIRED_RAW_COLUMNS)
    unexpected = tuple(column for column in df.columns if column not in normalized_or_raw)
    warnings: list[str] = []
    if unexpected:
        warnings.append("Unexpected columns are present and will be preserved by default.")
    return ValidationResult(
        ok=not missing,
        missing_columns=missing,
        unexpected_columns=unexpected,
        warnings=tuple(warnings),
    )


def validate_category_values(df: pd.DataFrame) -> ValidationResult:
    """Validate salary and department categories on normalized or raw columns."""

    data = normalize_column_names(df)
    invalid: dict[str, tuple[str, ...]] = {}
    if "salary" in data.columns:
        values = set(data["salary"].dropna().astype(str).unique())
        unexpected = tuple(sorted(values - set(ALLOWED_SALARY_VALUES)))
        if unexpected:
            invalid["salary"] = unexpected
    if "department" in data.columns:
        values = set(data["department"].dropna().astype(str).unique())
        unexpected = tuple(sorted(values - set(ALLOWED_DEPARTMENT_VALUES)))
        if unexpected:
            invalid["department"] = unexpected
    return ValidationResult(ok=not invalid, invalid_categories=invalid)
