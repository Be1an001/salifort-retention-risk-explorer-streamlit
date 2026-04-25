"""Training helpers for the local/dev Salifort MLOps Mini-Lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from salifort_mlops.config import (
    DEFAULT_THRESHOLD,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    LAB_MODEL_CANDIDATES,
    LAB_MODELS_DIR,
    LAB_REPORTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_DIR,
    MODEL_MODES,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from salifort_mlops.evaluate import evaluate_probabilities, find_best_threshold
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
            "name": "logistic_regression_balanced",
            "family": "logistic_regression",
            "imbalance_strategy": "class_weight=balanced",
        },
        {
            "name": "random_forest_balanced",
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


def make_ohe() -> OneHotEncoder:
    """Create a OneHotEncoder across sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessors(
    feature_columns: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict[str, ColumnTransformer]:
    """Build linear and tree preprocessing plans."""

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), categorical_features),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), categorical_features),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ],
        remainder="drop",
    )
    return {"linear": linear_preprocessor, "tree": tree_preprocessor}


def make_candidate_models(
    *,
    feature_columns: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    """Create the stable Phase 3 candidate model set."""

    preprocessors = build_preprocessors(
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )
    return {
        "logistic_regression_balanced": Pipeline(
            [
                ("preprocess", preprocessors["linear"]),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("preprocess", preprocessors["tree"]),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        n_estimators=120,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "xgb_weighted": Pipeline(
            [
                ("preprocess", preprocessors["tree"]),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=120,
                        learning_rate=0.06,
                        max_depth=4,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def _split_xy(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = get_model_feature_columns(mode)
    missing = [column for column in feature_columns + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError("Training data is missing required columns: " + ", ".join(missing))
    return df[feature_columns].copy(), df[TARGET_COLUMN].copy(), feature_columns


def _model_parameters(model: Pipeline) -> dict[str, Any]:
    estimator = model.named_steps["model"]
    params = estimator.get_params()
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def train_candidate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    mode: str = "operational",
) -> dict[str, Any]:
    """Train all lab candidates and evaluate reference and tuned thresholds."""

    mode = _validate_mode(mode)
    X_train, y_train, feature_columns = _split_xy(train_df, mode)
    X_test, y_test, _ = _split_xy(test_df, mode)
    categorical_features = ["department"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    if positive_count == 0:
        raise ValueError("Cannot train xgb_weighted because the training target has no positives.")
    scale_pos_weight = negative_count / positive_count

    candidates = make_candidate_models(
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scale_pos_weight=scale_pos_weight,
    )
    rows: list[dict[str, Any]] = []
    threshold_tables: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        probabilities = model.predict_proba(X_test)[:, 1]
        reference_metrics = evaluate_probabilities(
            y_test,
            probabilities,
            threshold=DEFAULT_THRESHOLD,
            fn_cost=FALSE_NEGATIVE_COST,
            fp_cost=FALSE_POSITIVE_COST,
        )
        best_metrics, threshold_table = find_best_threshold(
            y_test,
            probabilities,
            fn_cost=FALSE_NEGATIVE_COST,
            fp_cost=FALSE_POSITIVE_COST,
        )
        threshold_tables[model_name] = threshold_table.assign(model_name=model_name)
        rows.append(
            {
                "model_name": model_name,
                "mode": mode,
                "reference_threshold": DEFAULT_THRESHOLD,
                **{f"reference_{key}": value for key, value in reference_metrics.items()},
                **{f"best_{key}": value for key, value in best_metrics.items()},
                "scale_pos_weight": scale_pos_weight if model_name == "xgb_weighted" else None,
                "replaces_public_model": False,
            }
        )

    results_df = pd.DataFrame(rows)
    champion = select_lab_champion(results_df)
    return {
        "mode": mode,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "candidate_models": candidates,
        "fitted_models": fitted_models,
        "results_df": results_df,
        "threshold_tables": threshold_tables,
        "champion": champion,
    }


def select_lab_champion(results_df: pd.DataFrame) -> dict[str, Any]:
    """Select the lab champion without changing public model truth."""

    required = {"model_name", "best_cost", "best_recall", "best_f2", "best_precision"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError("Cannot select lab champion; missing columns: " + ", ".join(sorted(missing)))
    row = (
        results_df.sort_values(
            by=["best_cost", "best_recall", "best_f2", "best_precision"],
            ascending=[True, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
    row["replaces_public_model"] = False
    return row


def save_model_artifact(model: Pipeline, path: str | Path) -> Path:
    """Save a lab-only model artifact with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def save_training_reports(
    *,
    training_result: dict[str, Any],
    output_dir: Path = LAB_REPORTS_DIR,
) -> dict[str, Path]:
    """Save lab-only training result reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = training_result["results_df"]
    csv_path = output_dir / "training_results.csv"
    json_path = output_dir / "training_results.json"
    threshold_path = output_dir / "threshold_search_results.csv"
    results_df.to_csv(csv_path, index=False)
    pd.concat(training_result["threshold_tables"].values(), ignore_index=True).to_csv(
        threshold_path,
        index=False,
    )
    _write_json(
        {
            "mode": training_result["mode"],
            "feature_columns": training_result["feature_columns"],
            "candidate_models": list(training_result["fitted_models"].keys()),
            "lab_champion": training_result["champion"],
            "public_model_replacement": False,
            "public_reference_threshold": DEFAULT_THRESHOLD,
        },
        json_path,
    )
    return {
        "training_results_csv": csv_path,
        "training_results_json": json_path,
        "threshold_search_csv": threshold_path,
    }


def log_training_to_mlflow(
    *,
    training_result: dict[str, Any],
    model_paths: dict[str, Path],
) -> dict[str, Any]:
    """Log candidate params, metrics, tags, and joblib artifacts to local MLflow."""

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_summaries = []
    for model_name, model in training_result["fitted_models"].items():
        row = training_result["results_df"].loc[
            training_result["results_df"]["model_name"] == model_name
        ].iloc[0]
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tags(
                {
                    "project": "salifort-retention-risk-explorer",
                    "scope": "mlops-mini-lab",
                    "replaces_public_model": "false",
                    "mode": training_result["mode"],
                }
            )
            mlflow.log_params(_model_parameters(model))
            for key, value in row.to_dict().items():
                if key.startswith(("reference_", "best_")) and isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            mlflow.log_param("candidate_model", model_name)
            mlflow.log_param("public_reference_threshold", DEFAULT_THRESHOLD)
            mlflow.log_artifact(str(model_paths[model_name]))
            run_summaries.append(
                {
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "artifact_uri": run.info.artifact_uri,
                }
            )
    summary = {
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "tracking_uri": MLFLOW_TRACKING_DIR.as_uri(),
        "runs": run_summaries,
    }
    _write_json(summary, LAB_REPORTS_DIR / "mlflow_summary.json")
    return summary


def save_candidate_models(training_result: dict[str, Any]) -> dict[str, Path]:
    """Save every fitted candidate model under ``mlops/models``."""

    LAB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for model_name, model in training_result["fitted_models"].items():
        paths[model_name] = save_model_artifact(model, LAB_MODELS_DIR / f"{model_name}.joblib")
    return paths


def train_placeholder_not_implemented() -> None:
    """Retained for callers that explicitly request a placeholder."""

    raise NotImplementedError(
        "Use train_candidate_models for the local/dev MLOps lab. "
        "Training must remain outside the Streamlit runtime."
    )
