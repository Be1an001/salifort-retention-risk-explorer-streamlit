from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "hr_capstone_dataset.csv"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "v2"

COLUMN_RENAMES = {
    "Work_accident": "work_accident",
    "average_montly_hours": "average_monthly_hours",
    "time_spend_company": "tenure",
    "Department": "department",
}

FIGURE_PATHS = {
    "01_hours_vs_satisfaction_density": FIGURES_DIR / "01_hours-vs-satisfaction-density.png",
    "02_department_salary_attrition_promotion_heatmaps": FIGURES_DIR
    / "02_department-salary-attrition-promotion-heatmaps.png",
    "03_department_salary_count_heatmap": FIGURES_DIR / "03_department-salary-count-heatmap.png",
    "04_salary_retention_survival_like_curve": FIGURES_DIR
    / "04_salary-retention-survival-like-curve.png",
    "05_project_tenure_count_heatmap": FIGURES_DIR / "05_project-tenure-count-heatmap.png",
    "06_project_tenure_attrition_heatmap": FIGURES_DIR
    / "06_project-tenure-attrition-heatmap.png",
    "07_validation_metric_comparison": FIGURES_DIR / "07_validation-metric-comparison.png",
    "08_validation_pr_curves": FIGURES_DIR / "08_validation-pr-curves.png",
    "09_threshold_tuning_xgb_weighted": FIGURES_DIR / "09_threshold-tuning-xgb-weighted.png",
    "10_champion_confusion_matrix": FIGURES_DIR / "10_champion-confusion-matrix.png",
    "11_department_exposure_total_normalized": FIGURES_DIR
    / "11_department-exposure-total-normalized.png",
    "12_shap_summary_plot": FIGURES_DIR / "12_shap-summary-plot.png",
    "13_shap_dependence_number_project": FIGURES_DIR
    / "13_shap-dependence-number-project.png",
    "14_exec_summary_overview": FIGURES_DIR / "14_exec-summary-overview.png",
    "15_exec_summary_decision_threshold": FIGURES_DIR
    / "15_exec-summary-decision-threshold.png",
    "16_exec_summary_shap": FIGURES_DIR / "16_exec-summary-shap.png",
}

V2_ARTIFACT_PATHS = {
    "employee_scores": ARTIFACTS_ROOT / "employee_scores.parquet",
    "department_exposure": ARTIFACTS_ROOT / "department_exposure.csv",
    "threshold_curve": ARTIFACTS_ROOT / "threshold_curve.csv",
    "validation_model_comparison": ARTIFACTS_ROOT / "validation_model_comparison.csv",
    "confusion_matrix_at_selected_threshold": ARTIFACTS_ROOT
    / "confusion_matrix_at_selected_threshold.csv",
    "shap_importance": ARTIFACTS_ROOT / "shap_importance.csv",
    "metadata": ARTIFACTS_ROOT / "metadata.json",
    "employee_shap_sample": ARTIFACTS_ROOT / "employee_shap_sample.parquet",
    "pr_curve_points": ARTIFACTS_ROOT / "pr_curve_points.parquet",
    "model_modes_summary": ARTIFACTS_ROOT / "model_modes_summary.json",
}

V2_REQUIRED_ARTIFACTS = (
    "employee_scores",
    "department_exposure",
    "threshold_curve",
    "validation_model_comparison",
    "confusion_matrix_at_selected_threshold",
    "shap_importance",
    "metadata",
)

V2_ROW_KEY_SOURCE_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    "left",
    "promotion_last_5years",
    "department",
    "salary",
]


def get_repo_root() -> Path:
    return REPO_ROOT


def get_artifacts_root() -> Path:
    return ARTIFACTS_ROOT


def get_figure_paths() -> dict[str, Path]:
    return FIGURE_PATHS.copy()


def get_v2_artifact_paths() -> dict[str, Path]:
    return V2_ARTIFACT_PATHS.copy()


def get_v2_artifact_status() -> dict[str, bool]:
    return {name: path.exists() for name, path in V2_ARTIFACT_PATHS.items()}


def v2_artifacts_available() -> bool:
    return any(get_v2_artifact_status().values())


def v2_required_artifacts_complete() -> bool:
    status = get_v2_artifact_status()
    return all(status[name] for name in V2_REQUIRED_ARTIFACTS)


def _normalize_row_key_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def build_v2_row_key(df: pd.DataFrame) -> pd.Series:
    """Build a deterministic row key from cleaned row content for future artifact joins."""

    canonical_rows = df[V2_ROW_KEY_SOURCE_COLUMNS].apply(
        lambda row: "|".join(
            f"{column}={_normalize_row_key_value(row[column])}" for column in V2_ROW_KEY_SOURCE_COLUMNS
        ),
        axis=1,
    )
    return canonical_rows.map(
        lambda payload: f"smv2_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"
    )


def add_v2_row_identity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["employee_id_v2"] = build_v2_row_key(df)
    return df


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAMES).drop_duplicates().copy()
    df = df.reset_index(drop=True)
    df["salary"] = pd.Categorical(df["salary"], categories=["low", "medium", "high"], ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_app_data() -> pd.DataFrame:
    df = load_clean_data().copy()
    df = add_v2_row_identity(df)

    salary_as_text = df["salary"].astype(str)
    salary_weight_map = {"low": 1.00, "medium": 1.25, "high": 1.60}
    salary_score_map = {"low": 6.0, "medium": 3.0, "high": 0.0}

    df["employee_id"] = [f"SM-{index:05d}" for index in range(1, len(df) + 1)]
    df["attrition_label"] = df["left"].map({0: "Retained", 1: "Exited"})
    df["tenure_band"] = pd.Categorical(
        pd.cut(
            df["tenure"],
            bins=[0, 2, 4, 6, float("inf")],
            labels=["0-2 years", "3-4 years", "5-6 years", "7+ years"],
            include_lowest=True,
        ),
        categories=["0-2 years", "3-4 years", "5-6 years", "7+ years"],
        ordered=True,
    )
    df["overworked"] = (
        df["average_monthly_hours"].ge(220) | df["number_project"].ge(6)
    ).astype(int)
    df["project_intensity"] = pd.Categorical(
        pd.cut(
            df["number_project"],
            bins=[0, 2, 4, 6, float("inf")],
            labels=["Lean", "Balanced", "Stretch", "Extreme"],
            include_lowest=True,
        ),
        categories=["Lean", "Balanced", "Stretch", "Extreme"],
        ordered=True,
    )
    df["career_stall_flag"] = (
        df["promotion_last_5years"].eq(0) & df["tenure"].ge(5)
    ).astype(int)
    df["undervalued_flag"] = (
        salary_as_text.eq("low") & df["last_evaluation"].ge(0.70)
    ).astype(int)
    df["tenure_x_projects"] = df["tenure"] * df["number_project"]

    score = (
        (1 - df["satisfaction_level"].clip(lower=0, upper=1)) * 45
        + ((df["average_monthly_hours"] - 200).clip(lower=0) / 100).clip(upper=1) * 18
        + ((df["number_project"] - 4).clip(lower=0) / 3).clip(upper=1) * 12
        + ((df["tenure"] - 3).clip(lower=0) / 5).clip(upper=1) * 10
        + df["promotion_last_5years"].eq(0).astype(int) * 8
        + salary_as_text.map(salary_score_map).fillna(0)
        + df["last_evaluation"].ge(0.80).astype(int) * 4
    ).clip(lower=0, upper=100)

    df["screening_score_v1"] = score.round(1)
    df["high_risk_flag_v1"] = df["screening_score_v1"].ge(60).astype(int)
    df["salary_cost_weight"] = salary_as_text.map(salary_weight_map).fillna(1.0)
    df["risk_cost_exposure_index_v1"] = (
        df["screening_score_v1"] * df["salary_cost_weight"]
    ).round(2)

    return df


@st.cache_data(show_spinner=False)
def load_v2_metadata() -> dict[str, Any] | None:
    return _load_optional_json(V2_ARTIFACT_PATHS["metadata"])


@st.cache_data(show_spinner=False)
def load_v2_employee_scores() -> pd.DataFrame | None:
    return _load_optional_parquet(V2_ARTIFACT_PATHS["employee_scores"])


@st.cache_data(show_spinner=False)
def load_v2_department_exposure() -> pd.DataFrame | None:
    return _load_optional_csv(V2_ARTIFACT_PATHS["department_exposure"])


@st.cache_data(show_spinner=False)
def load_v2_threshold_curve() -> pd.DataFrame | None:
    return _load_optional_csv(V2_ARTIFACT_PATHS["threshold_curve"])


@st.cache_data(show_spinner=False)
def load_v2_validation_model_comparison() -> pd.DataFrame | None:
    return _load_optional_csv(V2_ARTIFACT_PATHS["validation_model_comparison"])


@st.cache_data(show_spinner=False)
def load_v2_confusion_matrix() -> pd.DataFrame | None:
    return _load_optional_csv(V2_ARTIFACT_PATHS["confusion_matrix_at_selected_threshold"])


@st.cache_data(show_spinner=False)
def load_v2_shap_importance() -> pd.DataFrame | None:
    return _load_optional_csv(V2_ARTIFACT_PATHS["shap_importance"])


@st.cache_data(show_spinner=False)
def load_v2_employee_shap_sample() -> pd.DataFrame | None:
    return _load_optional_parquet(V2_ARTIFACT_PATHS["employee_shap_sample"])


@st.cache_data(show_spinner=False)
def load_v2_pr_curve_points() -> pd.DataFrame | None:
    return _load_optional_parquet(V2_ARTIFACT_PATHS["pr_curve_points"])


@st.cache_data(show_spinner=False)
def load_v2_model_modes_summary() -> dict[str, Any] | None:
    return _load_optional_json(V2_ARTIFACT_PATHS["model_modes_summary"])
