from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback for minimal environments
    px = None

from app.utils.load_data import (
    get_runtime_mode_label,
    load_app_data,
    load_v2_department_exposure,
    load_v2_employee_scores,
)

V2_EMPLOYEE_REQUIRED_COLUMNS = {
    "employee_id",
    "department",
    "salary",
    "tenure",
    "tenure_band",
    "number_project",
    "project_intensity",
    "average_monthly_hours",
    "promotion_last_5years",
    "left",
    "attrition_label",
    "attrition_probability",
    "high_risk_flag",
    "risk_cost_exposure_index",
}

V2_DEPARTMENT_REQUIRED_COLUMNS = {
    "department",
    "headcount",
    "observed_attrition_rate",
    "avg_predicted_attrition",
    "high_risk_employees",
    "exposure_index",
    "exposure_per_100_employees",
}


def _is_valid_v2_employee_scores(df: pd.DataFrame | None) -> bool:
    return df is not None and V2_EMPLOYEE_REQUIRED_COLUMNS.issubset(df.columns)


def _is_valid_v2_department_exposure(df: pd.DataFrame | None) -> bool:
    return df is not None and V2_DEPARTMENT_REQUIRED_COLUMNS.issubset(df.columns)


def _initialize_filters(df: pd.DataFrame) -> None:
    department_options = sorted(df["department"].astype(str).unique().tolist())
    salary_options = [str(value) for value in df["salary"].astype(str).unique().tolist()]
    tenure_options = [str(value) for value in df["tenure_band"].astype(str).unique().tolist()]

    st.session_state.setdefault("department_filter", department_options)
    st.session_state.setdefault("salary_filter", salary_options)
    st.session_state.setdefault("tenure_band_filter", tenure_options)
    st.session_state.setdefault("high_risk_only", False)


def _apply_filters(df: pd.DataFrame, high_risk_column: str) -> pd.DataFrame:
    filtered = df[
        df["department"].astype(str).isin(st.session_state.department_filter)
        & df["salary"].astype(str).isin(st.session_state.salary_filter)
        & df["tenure_band"].astype(str).isin(st.session_state.tenure_band_filter)
    ].copy()

    if st.session_state.high_risk_only:
        filtered = filtered[filtered[high_risk_column].eq(1)].copy()

    return filtered


def _build_v1_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Department",
                "Headcount",
                "Observed Attrition Rate",
                "Avg Screening Score",
                "High-Risk Count",
                "Avg Exposure Index",
                "Total Exposure Index",
            ]
        )

    summary = (
        df.groupby("department", observed=True)
        .agg(
            headcount=("employee_id", "count"),
            observed_attrition_rate=("left", "mean"),
            average_screening_score=("screening_score_v1", "mean"),
            high_risk_count=("high_risk_flag_v1", "sum"),
            average_exposure_index=("risk_cost_exposure_index_v1", "mean"),
            total_exposure_index=("risk_cost_exposure_index_v1", "sum"),
        )
        .reset_index()
        .sort_values(["total_exposure_index", "headcount"], ascending=[False, False])
    )

    summary["observed_attrition_rate"] = summary["observed_attrition_rate"].mul(100)

    return summary.rename(
        columns={
            "department": "Department",
            "headcount": "Headcount",
            "observed_attrition_rate": "Observed Attrition Rate",
            "average_screening_score": "Avg Screening Score",
            "high_risk_count": "High-Risk Count",
            "average_exposure_index": "Avg Exposure Index",
            "total_exposure_index": "Total Exposure Index",
        }
    )


def _build_v2_department_summary_from_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Department",
                "Headcount",
                "Observed Attrition Rate",
                "Avg Predicted Attrition",
                "High-Risk Count",
                "Avg Exposure Index",
                "Total Exposure Index",
            ]
        )

    summary = (
        df.groupby("department", observed=True)
        .agg(
            headcount=("employee_id", "count"),
            observed_attrition_rate=("left", "mean"),
            avg_predicted_attrition=("attrition_probability", "mean"),
            high_risk_count=("high_risk_flag", "sum"),
            average_exposure_index=("risk_cost_exposure_index", "mean"),
            total_exposure_index=("risk_cost_exposure_index", "sum"),
        )
        .reset_index()
        .sort_values(["total_exposure_index", "headcount"], ascending=[False, False])
    )

    summary["observed_attrition_rate"] = summary["observed_attrition_rate"].mul(100)
    summary["avg_predicted_attrition"] = summary["avg_predicted_attrition"].mul(100)

    return summary.rename(
        columns={
            "department": "Department",
            "headcount": "Headcount",
            "observed_attrition_rate": "Observed Attrition Rate",
            "avg_predicted_attrition": "Avg Predicted Attrition",
            "high_risk_count": "High-Risk Count",
            "average_exposure_index": "Avg Exposure Index",
            "total_exposure_index": "Total Exposure Index",
        }
    )


def _prepare_v2_department_artifact_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.copy()
    summary = summary.rename(
        columns={
            "department": "Department",
            "headcount": "Headcount",
            "observed_attrition_rate": "Observed Attrition Rate",
            "avg_predicted_attrition": "Avg Predicted Attrition",
            "high_risk_employees": "High-Risk Count",
            "exposure_per_100_employees": "Avg Exposure Index",
            "exposure_index": "Total Exposure Index",
        }
    )

    summary["Observed Attrition Rate"] = summary["Observed Attrition Rate"].apply(
        lambda value: value * 100 if value <= 1 else value
    )
    summary["Avg Predicted Attrition"] = summary["Avg Predicted Attrition"].apply(
        lambda value: value * 100 if value <= 1 else value
    )

    return summary.sort_values(["Total Exposure Index", "Headcount"], ascending=[False, False])


def _can_use_department_artifact(df: pd.DataFrame) -> bool:
    all_salary_selected = set(st.session_state.salary_filter) == set(df["salary"].astype(str).unique())
    all_tenure_selected = set(st.session_state.tenure_band_filter) == set(
        df["tenure_band"].astype(str).unique()
    )
    return all_salary_selected and all_tenure_selected and not st.session_state.high_risk_only


def render() -> None:
    v1_df = load_app_data()
    v2_df = load_v2_employee_scores()
    v2_department_exposure = load_v2_department_exposure()

    use_v2_rows = _is_valid_v2_employee_scores(v2_df)
    base_df = v2_df if use_v2_rows else v1_df
    row_mode_label = "generated model outputs" if use_v2_rows else "screening score fallback"
    high_risk_column = "high_risk_flag" if use_v2_rows else "high_risk_flag_v1"
    risk_metric_label = "Avg Predicted Attrition" if use_v2_rows else "Avg Screening Score"

    _initialize_filters(base_df)

    st.title("Workforce Explorer")
    st.caption("Interactive workforce review built on the cleaned Salifort Motors dataset.")
    st.caption(f"Runtime mode: {get_runtime_mode_label()}. Row-level source: {row_mode_label}.")

    if use_v2_rows:
        st.info(
            "This page is using generated row-level model outputs when they are available. "
            "If a required file is missing or incomplete, it falls back to a simpler screening score for exploration."
        )
    else:
        st.warning(
            "The score shown on this page is a lightweight screening score for exploration only. "
            "It is not the final weighted XGBoost probability and should not be treated as the main model output."
        )

    with st.sidebar:
        st.subheader("Explorer Filters")
        st.multiselect(
            "Department",
            options=sorted(base_df["department"].astype(str).unique().tolist()),
            key="department_filter",
        )
        st.multiselect(
            "Salary",
            options=sorted(base_df["salary"].astype(str).unique().tolist()),
            key="salary_filter",
        )
        st.multiselect(
            "Tenure Band",
            options=sorted(base_df["tenure_band"].astype(str).unique().tolist()),
            key="tenure_band_filter",
        )
        st.checkbox(
            "Show selected-threshold employees only" if use_v2_rows else "Show high screening score only",
            key="high_risk_only",
        )

    filtered = _apply_filters(base_df, high_risk_column=high_risk_column)

    use_v2_department_artifact = use_v2_rows and _is_valid_v2_department_exposure(
        v2_department_exposure
    ) and _can_use_department_artifact(base_df)

    if use_v2_department_artifact:
        summary = _prepare_v2_department_artifact_summary(v2_department_exposure)
        summary = summary[summary["Department"].isin(st.session_state.department_filter)].copy()
        summary_source_label = "generated department exposure file"
    elif use_v2_rows:
        summary = _build_v2_department_summary_from_rows(filtered)
        summary_source_label = "row-level aggregation"
    else:
        summary = _build_v1_department_summary(filtered)
        summary_source_label = "screening-score aggregation"

    headcount = int(filtered.shape[0])
    observed_attrition_rate = float(filtered["left"].mean() * 100) if headcount else 0.0
    average_risk_metric = (
        float(filtered["attrition_probability"].mean() * 100)
        if use_v2_rows and headcount
        else float(filtered["screening_score_v1"].mean()) if headcount else 0.0
    )
    high_risk_count = int(filtered[high_risk_column].sum()) if headcount else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered Headcount", f"{headcount:,}")
    metric_cols[1].metric("Observed Attrition Rate", f"{observed_attrition_rate:.1f}%")
    metric_cols[2].metric(risk_metric_label, f"{average_risk_metric:.1f}" + ("%" if use_v2_rows else ""))
    metric_cols[3].metric("High-Risk Count", f"{high_risk_count:,}")

    st.caption(f"Department summary source: {summary_source_label}.")
    st.divider()

    chart_col, text_col = st.columns([1.15, 0.85], gap="large")

    with chart_col:
        st.subheader("Department Exposure Chart")
        if headcount == 0:
            st.info("No employees match the current filter combination.")
        elif px is not None:
            chart_data = summary.sort_values("Total Exposure Index", ascending=True)
            fig = px.bar(
                chart_data,
                x="Total Exposure Index",
                y="Department",
                orientation="h",
                color="Observed Attrition Rate",
                color_continuous_scale="Tealgrn",
                hover_data={
                    "Headcount": True,
                    risk_metric_label: ":.1f",
                    "High-Risk Count": True,
                    "Observed Attrition Rate": ":.1f",
                },
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar_title="Attrition %",
                xaxis_title="Risk-Cost Exposure Index",
                yaxis_title="Department",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(
                summary.set_index("Department")["Total Exposure Index"],
                use_container_width=True,
            )

    with text_col:
        st.subheader("How to Read This View")
        if use_v2_rows:
            st.markdown(
                "- This view is using generated row-level attrition probabilities.\n"
                "- The selected-threshold flag drives the flagged-employee filter and flagged count.\n"
                "- The exposure index comes from the generated risk-cost exposure field.\n"
                "- Department summaries use the generated exposure file when the current filters stay at the department level; otherwise they are rebuilt from the filtered row-level data."
            )
        else:
            st.markdown(
                "- The fallback score emphasizes low satisfaction, heavy workload, long tenure without promotion, and lower salary bands.\n"
                "- The fallback flag marks employees at or above the fallback screening cutoff for exploration.\n"
                "- The exposure index scales the fallback score by a simple salary cost weight to help compare screening concentration across teams.\n"
                "- Use this page to triage where to investigate, not to make case-level employment decisions."
            )

    st.subheader("Department Summary")
    summary_config: dict[str, object] = {
        "Observed Attrition Rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Avg Exposure Index": st.column_config.NumberColumn(format="%.2f"),
    }
    if use_v2_rows:
        summary_config["Avg Predicted Attrition"] = st.column_config.NumberColumn(format="%.1f%%")
    else:
        summary_config["Avg Screening Score"] = st.column_config.NumberColumn(format="%.1f")

    st.dataframe(
        summary.drop(columns=["Total Exposure Index"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
        column_config=summary_config,
    )

    st.subheader("Employee-Level Explorer Table")
    if use_v2_rows:
        selected_columns = [
            "employee_id",
            "department",
            "salary",
            "tenure",
            "tenure_band",
            "number_project",
            "project_intensity",
            "average_monthly_hours",
            "last_evaluation",
            "promotion_last_5years",
            "attrition_label",
            "attrition_probability",
            "high_risk_flag",
            "risk_cost_exposure_index",
        ]
        available_columns = [column for column in selected_columns if column in filtered.columns]
        employee_view = filtered[available_columns].rename(
            columns={
                "employee_id": "Employee ID",
                "department": "Department",
                "salary": "Salary",
                "tenure": "Tenure",
                "tenure_band": "Tenure Band",
                "number_project": "Projects",
                "project_intensity": "Project Intensity",
                "average_monthly_hours": "Avg Monthly Hours",
                "last_evaluation": "Last Evaluation",
                "promotion_last_5years": "Promoted In Last 5 Years",
                "attrition_label": "Observed Outcome",
                "attrition_probability": "Predicted Attrition",
                "high_risk_flag": "Flagged at Selected Threshold",
                "risk_cost_exposure_index": "Risk-Cost Exposure Index",
            }
        )
        table_config = {
            "Last Evaluation": st.column_config.NumberColumn(format="%.2f"),
            "Predicted Attrition": st.column_config.NumberColumn(format="%.3f"),
            "Risk-Cost Exposure Index": st.column_config.NumberColumn(format="%.2f"),
        }
    else:
        employee_view = filtered[
            [
                "employee_id",
                "department",
                "salary",
                "tenure",
                "tenure_band",
                "number_project",
                "project_intensity",
                "average_monthly_hours",
                "satisfaction_level",
                "promotion_last_5years",
                "attrition_label",
                "screening_score_v1",
                "high_risk_flag_v1",
                "risk_cost_exposure_index_v1",
            ]
        ].rename(
            columns={
                "employee_id": "Employee ID",
                "department": "Department",
                "salary": "Salary",
                "tenure": "Tenure",
                "tenure_band": "Tenure Band",
                "number_project": "Projects",
                "project_intensity": "Project Intensity",
                "average_monthly_hours": "Avg Monthly Hours",
                "satisfaction_level": "Satisfaction",
                "promotion_last_5years": "Promoted In Last 5 Years",
                "attrition_label": "Observed Outcome",
                "screening_score_v1": "Screening Score",
                "high_risk_flag_v1": "Fallback Flag",
                "risk_cost_exposure_index_v1": "Risk-Cost Exposure Index",
            }
        )
        table_config = {
            "Satisfaction": st.column_config.NumberColumn(format="%.2f"),
            "Screening Score": st.column_config.NumberColumn(format="%.1f"),
            "Risk-Cost Exposure Index": st.column_config.NumberColumn(format="%.2f"),
        }

    st.dataframe(
        employee_view,
        use_container_width=True,
        hide_index=True,
        column_config=table_config,
    )

    csv_buffer = StringIO()
    employee_view.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download filtered explorer CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="salifort_workforce_explorer_filtered.csv",
        mime="text/csv",
    )
