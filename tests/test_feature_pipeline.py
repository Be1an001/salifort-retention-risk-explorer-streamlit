from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from salifort_mlops.data_prep import clean_salifort_data
from salifort_mlops.features import (
    build_hr_features,
    get_model_feature_columns,
    split_features_target,
)


RAW_CSV = REPO_ROOT / "data" / "hr_capstone_dataset.csv"
ENGINEERED_COLUMNS = {
    "salary_level",
    "overworked",
    "project_intensity",
    "career_stall_flag",
    "undervalued_flag",
    "tenure_x_projects",
}


def _clean_data() -> pd.DataFrame:
    return clean_salifort_data(pd.read_csv(RAW_CSV))


def test_build_hr_features_adds_required_columns() -> None:
    features = build_hr_features(_clean_data(), mode="operational")
    assert ENGINEERED_COLUMNS.issubset(features.columns)
    assert features["project_intensity"].map(math.isfinite).all()


def test_salary_level_mapping_and_operational_feature_columns() -> None:
    features = build_hr_features(_clean_data(), mode="operational")
    salary_map = (
        features[["salary", "salary_level"]]
        .drop_duplicates()
        .sort_values("salary")
        .set_index("salary")["salary_level"]
        .to_dict()
    )
    assert salary_map == {"high": 2, "low": 0, "medium": 1}
    assert "satisfaction_level" not in get_model_feature_columns("operational")


def test_split_features_target_alignment() -> None:
    X, y = split_features_target(_clean_data(), mode="operational")
    assert len(X) == len(y) == 11991
    assert "left" not in X.columns
