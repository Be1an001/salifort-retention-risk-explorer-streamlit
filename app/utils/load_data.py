from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "hr_capstone_dataset.csv"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

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


def get_repo_root() -> Path:
    return REPO_ROOT


def get_figure_paths() -> dict[str, Path]:
    return FIGURE_PATHS.copy()


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAMES).drop_duplicates().copy()
    df = df.reset_index(drop=True)
    df["salary"] = pd.Categorical(df["salary"], categories=["low", "medium", "high"], ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_app_data() -> pd.DataFrame:
    df = load_clean_data().copy()

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
