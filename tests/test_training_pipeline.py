from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salifort_mlops.features import get_model_feature_columns
from salifort_mlops.train import (
    build_preprocessing_plan,
    describe_candidate_models,
    make_candidate_models,
    select_lab_champion,
)


def test_candidate_model_specs_are_stable() -> None:
    names = [item["name"] for item in describe_candidate_models()]
    assert names == [
        "logistic_regression_balanced",
        "random_forest_balanced",
        "xgb_weighted",
    ]


def test_make_candidate_models_contains_expected_candidates() -> None:
    feature_columns = get_model_feature_columns("operational")
    categorical = ["department"]
    numeric = [column for column in feature_columns if column not in categorical]
    candidates = make_candidate_models(
        feature_columns=feature_columns,
        categorical_features=categorical,
        numeric_features=numeric,
        scale_pos_weight=5.0,
    )
    assert set(candidates) == {
        "logistic_regression_balanced",
        "random_forest_balanced",
        "xgb_weighted",
    }


def test_select_lab_champion_ranking_logic() -> None:
    results = pd.DataFrame(
        [
            {
                "model_name": "a",
                "best_cost": 10,
                "best_recall": 0.9,
                "best_f2": 0.8,
                "best_precision": 0.7,
            },
            {
                "model_name": "b",
                "best_cost": 10,
                "best_recall": 0.95,
                "best_f2": 0.7,
                "best_precision": 0.6,
            },
            {
                "model_name": "c",
                "best_cost": 12,
                "best_recall": 1.0,
                "best_f2": 1.0,
                "best_precision": 1.0,
            },
        ]
    )
    champion = select_lab_champion(results)
    assert champion["model_name"] == "b"
    assert champion["replaces_public_model"] is False


def test_preprocessing_plan_excludes_operational_satisfaction() -> None:
    plan = build_preprocessing_plan("operational")
    assert "satisfaction_level" not in plan["numeric_features"]
