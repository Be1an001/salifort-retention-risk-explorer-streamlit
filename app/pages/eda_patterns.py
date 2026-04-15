from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_figure_paths


def _show_figure(figure_key: str, caption: str, interpretation: str) -> None:
    figures = get_figure_paths()
    st.image(str(figures[figure_key]), caption=caption, use_container_width=True)
    st.markdown(interpretation)


def render() -> None:
    st.title("EDA & Patterns")
    st.caption(
        "Selected project visuals from the analysis workflow, shown here as stable reference figures."
    )

    workload_tab, structure_tab, tenure_tab = st.tabs(
        ["Workload Patterns", "Department / Salary Structure", "Tenure / Project Structure"]
    )

    with workload_tab:
        _show_figure(
            "01_hours_vs_satisfaction_density",
            "Hours vs satisfaction density",
            "Low satisfaction clusters become more concerning when combined with sustained heavy monthly hours, which is one reason workload pressure matters operationally.",
        )
        _show_figure(
            "04_salary_retention_survival_like_curve",
            "Salary retention survival-like curve",
            "Retention patterns vary meaningfully across salary bands, reinforcing that compensation context matters when screening for exposure and planning intervention.",
        )

    with structure_tab:
        _show_figure(
            "02_department_salary_attrition_promotion_heatmaps",
            "Department, salary, attrition, and promotion heatmaps",
            "Department and pay structure do not move independently. Promotion access and observed attrition differ across combinations, which is useful when comparing pockets of workforce strain.",
        )
        _show_figure(
            "03_department_salary_count_heatmap",
            "Department and salary count heatmap",
            "The workforce is not evenly distributed. Larger departments can dominate total exposure, so raw counts and normalized rates should be read together.",
        )

    with tenure_tab:
        _show_figure(
            "05_project_tenure_count_heatmap",
            "Project count by tenure heatmap",
            "Project load and tenure form recognizable structural patterns, helping frame where employees may be carrying sustained delivery pressure.",
        )
        _show_figure(
            "06_project_tenure_attrition_heatmap",
            "Attrition heatmap across project count and tenure",
            "Higher observed attrition tends to concentrate in specific tenure and project-load combinations, which supports targeted early-warning screening rather than blanket action.",
        )
