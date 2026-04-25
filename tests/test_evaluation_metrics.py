from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salifort_mlops.evaluate import evaluate_probabilities, find_best_threshold


def test_evaluate_probabilities_known_case() -> None:
    metrics = evaluate_probabilities(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.7, 0.2],
        threshold=0.5,
        fn_cost=8.0,
        fp_cost=1.0,
    )
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["cost"] == 9.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_find_best_threshold_returns_required_fields() -> None:
    best, table = find_best_threshold([0, 0, 1, 1], [0.1, 0.4, 0.8, 0.9])
    assert 0.05 <= best["threshold"] <= 0.95
    for field in ["cost", "recall", "f2", "precision", "pr_auc", "roc_auc"]:
        assert field in best
    assert not table.empty


def test_cost_rule_prefers_false_positive_when_no_false_negative() -> None:
    metrics = evaluate_probabilities([0, 1], [0.6, 0.7], threshold=0.5)
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["cost"] == 1.0
