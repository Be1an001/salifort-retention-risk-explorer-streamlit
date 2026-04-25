"""Evaluation helpers for Salifort MLOps lab classification models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from salifort_mlops.config import (
    DEFAULT_THRESHOLD,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
)


def _safe_metric(metric_func: Any, *args: Any, default: float = 0.0, **kwargs: Any) -> float:
    try:
        value = metric_func(*args, **kwargs)
    except ValueError:
        return default
    if pd.isna(value):
        return default
    return float(value)


def classification_metrics_to_dict(
    *,
    y_true: Any,
    y_proba: Any,
    threshold: float,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> dict[str, float | int]:
    """Compute binary classification metrics for a probability threshold."""

    y_pred = (pd.Series(y_proba).astype(float) >= threshold).astype(int)
    y_true_series = pd.Series(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_series, y_pred, labels=[0, 1]).ravel()
    precision = _safe_metric(
        precision_score,
        y_true_series,
        y_pred,
        zero_division=0,
    )
    recall = _safe_metric(recall_score, y_true_series, y_pred, zero_division=0)
    f1 = _safe_metric(f1_score, y_true_series, y_pred, zero_division=0)
    f2 = _safe_metric(fbeta_score, y_true_series, y_pred, beta=2, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": _safe_metric(accuracy_score, y_true_series, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "roc_auc": _safe_metric(roc_auc_score, y_true_series, y_proba),
        "pr_auc": _safe_metric(average_precision_score, y_true_series, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost": float(false_negative_cost * fn + false_positive_cost * fp),
    }


def evaluate_probabilities(
    y_true: Any,
    y_proba: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
) -> dict[str, float | int]:
    """Evaluate probability scores with the lab threshold-cost rule."""

    return classification_metrics_to_dict(
        y_true=y_true,
        y_proba=y_proba,
        threshold=threshold,
        false_negative_cost=fn_cost,
        false_positive_cost=fp_cost,
    )


def find_best_threshold(
    y_true: Any,
    y_proba: Any,
    *,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Search thresholds by lowest cost, then recall, f2, and precision."""

    thresholds = [round(start + index * step, 10) for index in range(int((stop - start) / step) + 1)]
    rows = [
        evaluate_probabilities(
            y_true,
            y_proba,
            threshold=threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        for threshold in thresholds
    ]
    threshold_df = pd.DataFrame(rows)
    best_row = (
        threshold_df.sort_values(
            by=["cost", "recall", "f2", "precision"],
            ascending=[True, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
    return best_row, threshold_df


def binary_classification_metrics(
    y_true: Any,
    y_score: Any,
    *,
    threshold: float,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> dict[str, float | int]:
    """Backward-compatible alias for thresholded probability evaluation."""

    return classification_metrics_to_dict(
        y_true=y_true,
        y_proba=y_score,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )


def evaluate_placeholder_not_implemented() -> None:
    """Reserve full model evaluation for later phases."""

    raise NotImplementedError(
        "Full model evaluation reports are intentionally deferred beyond Phase 1."
    )
