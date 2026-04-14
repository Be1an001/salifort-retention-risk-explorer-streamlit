from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.load_data import add_v2_row_identity, get_artifacts_root, load_clean_data

SEED = 42
FALSE_NEGATIVE_COST = 8.0
FALSE_POSITIVE_COST = 1.0
DEFAULT_MODEL_MODE = "operational"
SALARY_COST_INDEX = {
    "low": 1.0,
    "medium": 1.3,
    "high": 1.7,
}

SOURCE_REPO_CANDIDATES = [
    REPO_ROOT.parent / "salifort-motors-attrition-modeling-python",
]
SOURCE_SCRIPT_RELATIVE = Path("scripts/02_salifort_motors_capstone_portfolio_project.py")
PUBLIC_REFERENCE_SELECTION = {
    "operational": {
        "model_name": "xgb_weighted",
        "selected_threshold": 0.29,
        "selection_basis": (
            "Public operational source-of-truth in the sibling repo README and walkthrough "
            "frames the final public version as weighted XGBoost at threshold 0.29."
        ),
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build precomputed V2 artifacts for the Salifort Streamlit app from the "
            "trusted offline modeling workflow."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_artifacts_root(),
        help="Directory where V2 artifacts should be written.",
    )
    parser.add_argument(
        "--model-mode",
        default=DEFAULT_MODEL_MODE,
        choices=["operational", "survey_rich"],
        help="Model mode to build from the trusted workflow. Public app default is operational.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the workflow without writing artifacts.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP artifact generation even if the dependency is available.",
    )
    return parser.parse_args()


def discover_source_workflow() -> dict[str, Any]:
    for repo_path in SOURCE_REPO_CANDIDATES:
        script_path = repo_path / SOURCE_SCRIPT_RELATIVE
        if repo_path.exists() and script_path.exists():
            return {
                "found": True,
                "repo_path": repo_path,
                "script_path": script_path,
                "description": (
                    "Sibling modeling repo with the exported weighted XGBoost, threshold, "
                    "and SHAP workflow used as the offline source of truth."
                ),
            }

    return {
        "found": False,
        "repo_path": None,
        "script_path": None,
        "description": (
            "No trusted sibling modeling workflow was found. The builder can stay as "
            "scaffolding only until the source workflow is available."
        ),
    }


def load_builder_dependencies(skip_shap: bool) -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            fbeta_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "Missing offline builder dependency: "
            f"{exc}. Install the modeling requirements from the sibling source repo "
            "before running this builder."
        ) from exc

    shap_module = None
    shap_status = "skipped" if skip_shap else "not_requested"
    shap_error = None
    if not skip_shap:
        try:
            import shap as shap_module
            shap_status = "available"
        except Exception as exc:
            shap_module = None
            shap_status = "import_failed"
            shap_error = f"{type(exc).__name__}: {exc}"

    return {
        "np": np,
        "pd": pd,
        "sklearn": sklearn,
        "SMOTE": SMOTE,
        "ImbPipeline": ImbPipeline,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "fbeta_score": fbeta_score,
        "precision_recall_curve": precision_recall_curve,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "XGBClassifier": XGBClassifier,
        "shap": shap_module,
        "shap_status": shap_status,
        "shap_error": shap_error,
    }


def make_ohe(OneHotEncoder: Any) -> Any:
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_xgb(XGBClassifier: Any, scale_pos_weight: float, seed: int = SEED) -> Any:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=seed,
        tree_method="hist",
    )


def derive_tenure_band(pd: Any, tenure: Any) -> Any:
    return pd.Categorical(
        pd.cut(
            tenure,
            bins=[0, 2, 4, 6, float("inf")],
            labels=["0-2 years", "3-4 years", "5-6 years", "7+ years"],
            include_lowest=True,
        ),
        categories=["0-2 years", "3-4 years", "5-6 years", "7+ years"],
        ordered=True,
    )


def build_hr_features(df: Any, np: Any, pd: Any, mode: str = DEFAULT_MODEL_MODE) -> Any:
    data = df.copy()

    salary_as_text = data["salary"].astype(str)
    data["salary_level"] = salary_as_text.map({"low": 0, "medium": 1, "high": 2})
    data["tenure_band"] = derive_tenure_band(pd, data["tenure"])
    data["overworked"] = (data["average_monthly_hours"] > 175).astype(int)
    data["project_intensity"] = (
        data["average_monthly_hours"]
        / data["number_project"].replace({0: np.nan})
    ).fillna(0)
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
            (data["satisfaction_level"] < 0.45)
            & (data["last_evaluation"] > 0.75)
        ).astype(int)

    return data


def build_model_frame(feature_df: Any, mode: str) -> tuple[Any, list[str], list[str], list[str]]:
    drop_cols = ["salary", "employee_id_v2", "tenure_band", "attrition_label"]
    if mode == "operational":
        for column in [
            "satisfaction_level",
            "burnout_index",
            "effort_reward_gap",
            "low_satisfaction_high_eval",
        ]:
            if column in feature_df.columns:
                drop_cols.append(column)

    model_df = feature_df.drop(columns=[column for column in drop_cols if column in feature_df.columns])
    categorical_features = ["department"]
    feature_cols = [column for column in model_df.columns if column != "left"]
    numeric_features = [
        column for column in feature_cols if column not in categorical_features
    ]
    return model_df, categorical_features, feature_cols, numeric_features


def evaluate_probabilities(
    y_true: Any,
    y_proba: Any,
    threshold: float,
    deps: dict[str, Any],
) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = deps["confusion_matrix"](y_true, y_pred, labels=[0, 1]).ravel()
    flagged_count = int(y_pred.sum())
    total_count = len(y_pred)

    return {
        "threshold": float(threshold),
        "accuracy": float(deps["accuracy_score"](y_true, y_pred)),
        "precision": float(deps["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(deps["recall_score"](y_true, y_pred, zero_division=0)),
        "f1": float(deps["f1_score"](y_true, y_pred, zero_division=0)),
        "f2": float(deps["fbeta_score"](y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(deps["roc_auc_score"](y_true, y_proba)),
        "pr_auc": float(deps["average_precision_score"](y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost": float(FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp),
        "flagged_count": flagged_count,
        "flagged_rate": float(flagged_count / total_count if total_count else 0.0),
    }


def find_best_threshold(y_true: Any, y_proba: Any, deps: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    rows = []
    for threshold in deps["np"].arange(0.05, 0.96, 0.01):
        rows.append(evaluate_probabilities(y_true, y_proba, float(threshold), deps))

    threshold_df = deps["pd"].DataFrame(rows)
    best_row = (
        threshold_df.sort_values(
            by=["cost", "recall", "f2", "precision"],
            ascending=[True, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
    return best_row, threshold_df


def build_model_specs(
    deps: dict[str, Any],
    categorical_features: list[str],
    numeric_features: list[str],
    neg_pos_ratio: float,
) -> dict[str, Any]:
    linear_preprocessor = deps["ColumnTransformer"](
        transformers=[
            ("cat", make_ohe(deps["OneHotEncoder"]), categorical_features),
            (
                "num",
                deps["Pipeline"](
                    [
                        ("imputer", deps["SimpleImputer"](strategy="median")),
                        ("scaler", deps["StandardScaler"]()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    tree_preprocessor = deps["ColumnTransformer"](
        transformers=[
            ("cat", make_ohe(deps["OneHotEncoder"]), categorical_features),
            (
                "num",
                deps["Pipeline"](
                    [("imputer", deps["SimpleImputer"](strategy="median"))]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    return {
        "log_reg_balanced": deps["ImbPipeline"](
            [
                ("preprocess", linear_preprocessor),
                (
                    "model",
                    deps["LogisticRegression"](
                        max_iter=3000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "log_reg_smote": deps["ImbPipeline"](
            [
                ("preprocess", linear_preprocessor),
                ("smote", deps["SMOTE"](random_state=SEED)),
                (
                    "model",
                    deps["LogisticRegression"](
                        max_iter=3000,
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "rf_balanced": deps["ImbPipeline"](
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    deps["RandomForestClassifier"](
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "xgb_weighted": deps["ImbPipeline"](
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    make_xgb(
                        deps["XGBClassifier"],
                        scale_pos_weight=neg_pos_ratio,
                        seed=SEED,
                    ),
                ),
            ]
        ),
    }


def fit_and_score_models(feature_df: Any, mode: str, deps: dict[str, Any]) -> dict[str, Any]:
    model_df, categorical_features, feature_cols, numeric_features = build_model_frame(
        feature_df, mode
    )

    X = model_df[feature_cols].copy()
    y = model_df["left"].copy()

    train_test_split = deps["train_test_split"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=SEED,
    )

    neg_pos_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    model_specs = build_model_specs(
        deps,
        categorical_features,
        numeric_features,
        neg_pos_ratio,
    )

    validation_rows: list[dict[str, Any]] = []
    threshold_tables: dict[str, Any] = {}
    validation_proba_store: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, pipeline in model_specs.items():
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        valid_proba = pipeline.predict_proba(X_valid)[:, 1]
        validation_proba_store[model_name] = valid_proba
        best_threshold_row, threshold_table = find_best_threshold(y_valid, valid_proba, deps)
        threshold_tables[model_name] = threshold_table

        base_metrics = evaluate_probabilities(y_valid, valid_proba, 0.50, deps)
        best_metrics = evaluate_probabilities(
            y_valid,
            valid_proba,
            float(best_threshold_row["threshold"]),
            deps,
        )

        validation_rows.append(
            {
                "model": model_name,
                "threshold_0_5_recall": base_metrics["recall"],
                "threshold_0_5_f1": base_metrics["f1"],
                "threshold_0_5_pr_auc": base_metrics["pr_auc"],
                "best_threshold": float(best_threshold_row["threshold"]),
                "best_recall": best_metrics["recall"],
                "best_precision": best_metrics["precision"],
                "best_f1": best_metrics["f1"],
                "best_f2": best_metrics["f2"],
                "best_roc_auc": best_metrics["roc_auc"],
                "best_pr_auc": best_metrics["pr_auc"],
                "best_cost": best_metrics["cost"],
                "model_mode": mode,
            }
        )

    validation_results = deps["pd"].DataFrame(validation_rows).sort_values(
        by=["best_cost", "best_recall", "best_f2"],
        ascending=[True, False, False],
    )

    metric_leader_name = str(validation_results.iloc[0]["model"])
    metric_leader_threshold = float(validation_results.iloc[0]["best_threshold"])

    public_reference = PUBLIC_REFERENCE_SELECTION.get(mode)
    if public_reference and public_reference["model_name"] in model_specs:
        selected_model_name = str(public_reference["model_name"])
        selected_threshold = float(public_reference["selected_threshold"])
        selection_mode = "public_reference"
        selection_note = public_reference["selection_basis"]
    else:
        selected_model_name = metric_leader_name
        selected_threshold = metric_leader_threshold
        selection_mode = "metric_leader"
        selection_note = "Selected directly from the validation metric leaderboard."

    selected_pipeline = model_specs[selected_model_name]
    selected_pipeline.fit(X_train_full, y_train_full)

    test_proba = selected_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_proba, selected_threshold, deps)

    full_proba = selected_pipeline.predict_proba(X)[:, 1]
    full_pred = (full_proba >= selected_threshold).astype(int)

    return {
        "X": X,
        "X_test": X_test,
        "y_valid": y_valid,
        "y": y,
        "y_test": y_test,
        "full_proba": full_proba,
        "full_pred": full_pred,
        "validation_results": validation_results,
        "threshold_table": threshold_tables[selected_model_name].assign(
            model_name=selected_model_name,
            model_mode=mode,
        ),
        "threshold_tables": threshold_tables,
        "validation_proba_store": validation_proba_store,
        "fitted_models": fitted_models,
        "metric_leader_name": metric_leader_name,
        "metric_leader_threshold": metric_leader_threshold,
        "selected_model_name": selected_model_name,
        "selected_threshold": selected_threshold,
        "selected_pipeline": selected_pipeline,
        "selection_mode": selection_mode,
        "selection_note": selection_note,
        "test_metrics": test_metrics,
    }


def build_employee_scores(clean_df: Any, feature_df: Any, fit_results: dict[str, Any], deps: dict[str, Any]) -> Any:
    report_df = clean_df.copy()
    report_df["salary_level"] = feature_df["salary_level"]
    report_df["attrition_label"] = report_df["left"].map({0: "Retained", 1: "Exited"})
    report_df["overworked"] = feature_df["overworked"]
    report_df["project_intensity"] = feature_df["project_intensity"]
    report_df["career_stall_flag"] = feature_df["career_stall_flag"]
    report_df["undervalued_flag"] = feature_df["undervalued_flag"]
    report_df["tenure_x_projects"] = feature_df["tenure_x_projects"]
    report_df["tenure_band"] = feature_df["tenure_band"]
    report_df["attrition_probability"] = fit_results["full_proba"]
    report_df["selected_threshold_flag"] = fit_results["full_pred"].astype(int)
    report_df["high_risk_flag"] = report_df["selected_threshold_flag"]
    report_df["risk_decile"] = (
        deps["pd"].qcut(
            report_df["attrition_probability"].rank(method="first"),
            q=10,
            labels=list(range(1, 11)),
        )
        .astype(int)
    )
    report_df["salary_cost_weight"] = report_df["salary"].astype(str).map(SALARY_COST_INDEX).fillna(1.0)
    report_df["risk_cost_exposure_index"] = (
        report_df["attrition_probability"] * report_df["salary_cost_weight"]
    )
    report_df["employee_id"] = report_df["employee_id_v2"]
    return report_df


def build_department_exposure(employee_scores: Any, selected_threshold: float, model_mode: str) -> Any:
    department_exposure = (
        employee_scores.groupby("department", observed=True)
        .agg(
            headcount=("left", "size"),
            observed_attrition_rate=("left", "mean"),
            avg_predicted_attrition=("attrition_probability", "mean"),
            high_risk_employees=("selected_threshold_flag", "sum"),
            exposure_index=("risk_cost_exposure_index", "sum"),
        )
        .reset_index()
        .sort_values("exposure_index", ascending=False)
    )
    department_exposure["exposure_per_100_employees"] = (
        department_exposure["exposure_index"] / department_exposure["headcount"] * 100
    )
    department_exposure["selected_threshold"] = selected_threshold
    department_exposure["model_mode"] = model_mode
    return department_exposure


def build_confusion_matrix_table(test_metrics: dict[str, Any], selected_threshold: float, pd: Any) -> Any:
    return pd.DataFrame(
        [
            {
                "selected_threshold": selected_threshold,
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "f2": test_metrics["f2"],
                "accuracy": test_metrics["accuracy"],
            }
        ]
    )


def build_pr_curve_points(
    validation_proba_store: dict[str, Any],
    y_valid: Any,
    model_mode: str,
    deps: dict[str, Any],
) -> Any:
    rows: list[dict[str, Any]] = []
    for model_name, probabilities in validation_proba_store.items():
        precision, recall, _ = deps["precision_recall_curve"](y_valid, probabilities)
        rows.extend(
            {
                "model": model_name,
                "recall": float(recall_value),
                "precision": float(precision_value),
                "model_mode": model_mode,
            }
            for precision_value, recall_value in zip(precision, recall)
        )
    return deps["pd"].DataFrame(rows)


def build_shap_outputs(
    fit_results: dict[str, Any],
    employee_scores: Any,
    model_mode: str,
    deps: dict[str, Any],
) -> tuple[Any | None, Any | None, str | None]:
    shap_module = deps["shap"]
    if shap_module is None:
        if deps.get("shap_status") == "skipped":
            return None, None, "SHAP generation was skipped by the --skip-shap flag."
        if deps.get("shap_error"):
            return None, None, f"SHAP import failed: {deps['shap_error']}"
        return None, None, "SHAP is not available in the current builder environment."

    explain_pipeline = fit_results["selected_pipeline"]
    preprocess = explain_pipeline.named_steps["preprocess"]
    explain_model = explain_pipeline.named_steps["model"]
    X_test = fit_results["X_test"]
    X_test_enc = preprocess.transform(X_test)
    feature_names = preprocess.get_feature_names_out()

    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "") for name in feature_names
    ]

    sample_size = min(800, X_test.shape[0])
    sample_index = deps["np"].random.choice(
        deps["np"].arange(X_test.shape[0]),
        size=sample_size,
        replace=False,
    )
    sampled_positions = X_test.index[sample_index]
    X_shap = X_test_enc[sample_index]
    X_shap_df = deps["pd"].DataFrame(X_shap, columns=clean_feature_names)

    try:
        explainer = shap_module.TreeExplainer(explain_model)
        shap_values = explainer.shap_values(X_shap)
    except Exception as exc:
        return None, None, f"SHAP generation failed: {type(exc).__name__}: {exc}"
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_importance = deps["pd"].DataFrame(
        {
            "feature": clean_feature_names,
            "mean_abs_shap": deps["np"].abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance["rank"] = range(1, len(shap_importance) + 1)
    shap_importance["model_name"] = fit_results["selected_model_name"]
    shap_importance["model_mode"] = model_mode

    shap_long = []
    sample_employee_ids = employee_scores.loc[sampled_positions, "employee_id_v2"].tolist()
    for row_number, employee_id in enumerate(sample_employee_ids):
        for feature_name, shap_value, feature_value in zip(
            clean_feature_names,
            shap_values[row_number],
            X_shap_df.iloc[row_number].tolist(),
        ):
            shap_long.append(
                {
                    "employee_id_v2": employee_id,
                    "feature": feature_name,
                    "shap_value": float(shap_value),
                    "feature_value": float(feature_value),
                    "model_mode": model_mode,
                }
            )

    return shap_importance, deps["pd"].DataFrame(shap_long), None


def build_metadata(
    clean_df: Any,
    fit_results: dict[str, Any],
    model_mode: str,
    source_workflow: dict[str, Any],
    raw_row_count: int,
) -> dict[str, Any]:
    raw_rows = raw_row_count
    clean_rows = int(clean_df.shape[0])
    duplicates_removed = raw_rows - clean_rows
    attrition_rate_clean = float(clean_df["left"].mean())
    champion_name = fit_results["selected_model_name"]
    final_model = {
        "xgb_weighted": "weighted XGBoost",
        "rf_balanced": "balanced random forest",
        "log_reg_balanced": "balanced logistic regression",
        "log_reg_smote": "SMOTE logistic regression",
    }.get(champion_name, champion_name)

    return {
        "project_name": "Salifort Motors Retention Risk Explorer",
        "project_subtitle": "Operational HR Analytics Decision App",
        "dataset_rows_raw": raw_rows,
        "duplicates_removed": duplicates_removed,
        "dataset_rows_clean": clean_rows,
        "attrition_rate_clean": attrition_rate_clean,
        "final_model": final_model,
        "selected_threshold": fit_results["selected_threshold"],
        "selected_test_recall": fit_results["test_metrics"]["recall"],
        "selected_test_precision": fit_results["test_metrics"]["precision"],
        "selected_test_accuracy": fit_results["test_metrics"]["accuracy"],
        "model_mode_main": model_mode,
        "artifact_build_date": datetime.now(timezone.utc).isoformat(),
        "artifact_version": f"v2-offline-{model_mode}",
        "notes": [
            "Built outside Streamlit runtime from the trusted exported modeling workflow.",
            f"Source workflow: {source_workflow['script_path']}" if source_workflow["script_path"] else source_workflow["description"],
            "Threshold selected using the false-negative-heavy cost rule from the source workflow.",
            fit_results["selection_note"],
            (
                f"Local rerun metric leader was {fit_results['metric_leader_name']} at "
                f"{fit_results['metric_leader_threshold']:.2f}."
            ),
        ],
    }


def build_model_modes_summary(source_workflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "operational": {
            "feature_inclusion_summary": (
                "Removes direct satisfaction-based modeling features and keeps the "
                "more deployment-like public workflow."
            ),
            "deployment_notes": (
                "This is the main public mode used by the Streamlit app and the "
                "preferred default for offline artifact generation."
            ),
        },
        "survey_rich": {
            "feature_inclusion_summary": (
                "Keeps stronger explanatory survey-driven features such as "
                "burnout_index and effort_reward_gap."
            ),
            "deployment_notes": (
                "Useful for comparison and interpretation, but not the main public "
                "deployment framing."
            ),
        },
        "comparison_notes": (
            "Both modes come from the trusted sibling modeling workflow. The "
            "operational mode is the main public version in this Streamlit repo."
        ),
        "source_workflow": str(source_workflow["script_path"]) if source_workflow["script_path"] else None,
    }


def write_dataframe(df: Any, path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:  # pragma: no cover - guarded by call sites
        raise ValueError(f"Unsupported dataframe output format: {path.suffix}")


def write_json(payload: dict[str, Any], path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    source_workflow = discover_source_workflow()
    if not source_workflow["found"]:
        raise SystemExit(source_workflow["description"])

    deps = load_builder_dependencies(skip_shap=args.skip_shap)
    pd = deps["pd"]

    clean_df = add_v2_row_identity(load_clean_data())
    raw_row_count = int(pd.read_csv(REPO_ROOT / "data" / "hr_capstone_dataset.csv").shape[0])
    feature_df = build_hr_features(clean_df, deps["np"], pd, mode=args.model_mode)
    fit_results = fit_and_score_models(feature_df, args.model_mode, deps)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    employee_scores = build_employee_scores(clean_df, feature_df, fit_results, deps)
    department_exposure = build_department_exposure(
        employee_scores,
        fit_results["selected_threshold"],
        args.model_mode,
    )
    confusion_matrix_df = build_confusion_matrix_table(
        fit_results["test_metrics"],
        fit_results["selected_threshold"],
        pd,
    )
    metadata = build_metadata(
        clean_df,
        fit_results,
        args.model_mode,
        source_workflow,
        raw_row_count,
    )
    pr_curve_points = build_pr_curve_points(
        fit_results["validation_proba_store"],
        fit_results["y_valid"],
        args.model_mode,
        deps,
    )

    threshold_curve = fit_results["threshold_table"].copy()
    threshold_curve["model_name"] = fit_results["selected_model_name"]
    threshold_curve["model_mode"] = args.model_mode

    validation_comparison = fit_results["validation_results"].copy()
    shap_importance, employee_shap_sample, shap_message = build_shap_outputs(
        fit_results,
        employee_scores,
        args.model_mode,
        deps,
    )

    write_json(metadata, output_root / "metadata.json", dry_run=args.dry_run)
    write_dataframe(employee_scores, output_root / "employee_scores.parquet", dry_run=args.dry_run)
    write_dataframe(department_exposure, output_root / "department_exposure.csv", dry_run=args.dry_run)
    write_dataframe(threshold_curve, output_root / "threshold_curve.csv", dry_run=args.dry_run)
    write_dataframe(
        validation_comparison,
        output_root / "validation_model_comparison.csv",
        dry_run=args.dry_run,
    )
    write_dataframe(
        confusion_matrix_df,
        output_root / "confusion_matrix_at_selected_threshold.csv",
        dry_run=args.dry_run,
    )

    if shap_importance is not None:
        write_dataframe(shap_importance, output_root / "shap_importance.csv", dry_run=args.dry_run)
    else:
        print(shap_message or "Skipping shap outputs because SHAP could not be generated.")

    if employee_shap_sample is not None:
        write_dataframe(
            employee_shap_sample,
            output_root / "employee_shap_sample.parquet",
            dry_run=args.dry_run,
        )

    write_dataframe(pr_curve_points, output_root / "pr_curve_points.parquet", dry_run=args.dry_run)
    write_json(
        build_model_modes_summary(source_workflow),
        output_root / "model_modes_summary.json",
        dry_run=args.dry_run,
    )

    print(f"Source workflow: {source_workflow['script_path']}")
    print(f"Metric leader on this rerun: {fit_results['metric_leader_name']}")
    print(f"Metric-leader threshold on this rerun: {fit_results['metric_leader_threshold']:.2f}")
    print(f"Selected model: {fit_results['selected_model_name']}")
    print(f"Selected threshold: {fit_results['selected_threshold']:.2f}")
    print(f"Selection mode: {fit_results['selection_mode']}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
