"""Dependency-light evaluation helpers for future lab models."""

from __future__ import annotations

from collections.abc import Sequence

from salifort_mlops.config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST


def confusion_counts(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float,
) -> dict[str, int]:
    """Return binary confusion counts using a probability threshold."""

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    tn = fp = fn = tp = 0
    for actual, score in zip(y_true, y_score):
        predicted = int(float(score) >= threshold)
        actual_int = int(actual)
        if actual_int == 0 and predicted == 0:
            tn += 1
        elif actual_int == 0 and predicted == 1:
            fp += 1
        elif actual_int == 1 and predicted == 0:
            fn += 1
        elif actual_int == 1 and predicted == 1:
            tp += 1
        else:
            raise ValueError("y_true values must be binary 0/1 labels.")
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def binary_classification_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> dict[str, float | int]:
    """Compute core threshold metrics without requiring sklearn."""

    counts = confusion_counts(y_true, y_score, threshold=threshold)
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta_squared = 4
    f2 = (
        (1 + beta_squared)
        * precision
        * recall
        / ((beta_squared * precision) + recall)
        if ((beta_squared * precision) + recall)
        else 0.0
    )
    cost = false_negative_cost * fn + false_positive_cost * fp
    return {
        "threshold": float(threshold),
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "cost": cost,
    }


def evaluate_placeholder_not_implemented() -> None:
    """Reserve full model evaluation for later phases."""

    raise NotImplementedError(
        "Full model evaluation reports are intentionally deferred beyond Phase 1."
    )
