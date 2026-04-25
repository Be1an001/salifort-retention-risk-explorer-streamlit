from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from salifort_mlops.config import ALLOWED_DEPARTMENT_VALUES, ALLOWED_SALARY_VALUES
from salifort_mlops.data_prep import clean_salifort_data, standardize_columns
from salifort_mlops.schemas import get_required_raw_columns


RAW_CSV = REPO_ROOT / "data" / "hr_capstone_dataset.csv"


def test_raw_csv_exists_and_has_required_columns() -> None:
    assert RAW_CSV.exists()
    raw = pd.read_csv(RAW_CSV)
    assert set(get_required_raw_columns()).issubset(raw.columns)


def test_standardize_columns_creates_normalized_names() -> None:
    raw = pd.read_csv(RAW_CSV)
    standardized = standardize_columns(raw)
    assert "average_monthly_hours" in standardized.columns
    assert "tenure" in standardized.columns
    assert "work_accident" in standardized.columns
    assert "department" in standardized.columns


def test_clean_data_contract_matches_current_dataset() -> None:
    raw = pd.read_csv(RAW_CSV)
    clean = clean_salifort_data(raw)
    assert len(clean) == 11991
    assert "left" in clean.columns
    assert set(clean["salary"].dropna().astype(str)).issubset(ALLOWED_SALARY_VALUES)
    assert set(clean["department"].dropna().astype(str)).issubset(
        ALLOWED_DEPARTMENT_VALUES
    )
