from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback for minimal environments
    px = None

from app.utils.load_data import load_app_data


def _initialize_filters(df: pd.DataFrame) -> None:
    department_options = sorted(df["department"].astype(str).unique().tolist())
    salary_options = list(df["salary"].cat.categories)
    tenure_options = list(df["tenure_band"].cat.categories)

    st.session_state.setdefault("department_filter", department_options)
    st.session_state.setdefault("salary_filter", salary_options)
    st.session_state.setdefault("tenure_band_filter", tenure_options)
    st.session_state.setdefault("high_risk_only", False)


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        df["department"].isin(st.session_state.department_filter)
        & df["salary"].astype(str).isin(st.session_state.salary_filter)
        & df["tenure_band"].astype(str).isin(st.session_state.tenure_band_filter)
    ].copy()

    if st.session_state.high_risk_only:
        filtered = filtered[filtered["high_risk_flag_v1"].eq(1)].copy()

    return filtered


def _build_department_summary(df: pd.DataFrame) -> pd.DataFrame:
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


def render() -> None:
    df = load_app_data()
    _initialize_filters(df)

    st.title("Workforce Explorer")
    st.caption("Interactive workforce screening view built on the cleaned Salifort Motors dataset.")

    st.warning(
        "The score shown on this page is a V1 app-facing screening proxy for exploration only. "
        "It is not the deployed weighted XGBoost probability and should not be interpreted as the production model output."
    )

    with st.sidebar:
        st.subheader("Explorer Filters")
        st.multiselect(
            "Department",
            options=sorted(df["department"].astype(str).unique().tolist()),
            key="department_filter",
        )
        st.multiselect(
            "Salary",
            options=list(df["salary"].cat.categories),
            key="salary_filter",
        )
        st.multiselect(
            "Tenure Band",
            options=list(df["tenure_band"].cat.categories),
            key="tenure_band_filter",
        )
        st.checkbox("Show high-risk proxy only", key="high_risk_only")

    filtered = _apply_filters(df)
    summary = _build_department_summary(filtered)

    headcount = int(filtered.shape[0])
    observed_attrition_rate = float(filtered["left"].mean() * 100) if headcount else 0.0
    average_score = float(filtered["screening_score_v1"].mean()) if headcount else 0.0
    high_risk_count = int(filtered["high_risk_flag_v1"].sum()) if headcount else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered Headcount", f"{headcount:,}")
    metric_cols[1].metric("Observed Attrition Rate", f"{observed_attrition_rate:.1f}%")
    metric_cols[2].metric("Average Screening Score", f"{average_score:.1f}")
    metric_cols[3].metric("High-Risk Count", f"{high_risk_count:,}")

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
                    "Avg Screening Score": ":.1f",
                    "High-Risk Count": True,
                    "Observed Attrition Rate": ":.1f",
                },
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar_title="Attrition %",
                xaxis_title="Total Risk-Cost Exposure Index (V1)",
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
        st.markdown(
            "- The proxy score emphasizes low satisfaction, heavy workload, long tenure without promotion, and lower salary bands.\n"
            "- `high_risk_flag_v1` marks employees at or above the V1 screening cutoff for exploration.\n"
            "- The exposure index scales the proxy score by a simple salary cost weight to help compare screening concentration across teams.\n"
            "- Use this page to triage where to investigate, not to make case-level employment decisions."
        )

    st.subheader("Department Summary")
    st.dataframe(
        summary.drop(columns=["Total Exposure Index"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Observed Attrition Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg Screening Score": st.column_config.NumberColumn(format="%.1f"),
            "Avg Exposure Index": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.subheader("Employee-Level Explorer Table")
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
            "screening_score_v1": "V1 Screening Score",
            "high_risk_flag_v1": "High-Risk Proxy",
            "risk_cost_exposure_index_v1": "Risk-Cost Exposure Index",
        }
    )

    st.dataframe(
        employee_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Satisfaction": st.column_config.NumberColumn(format="%.2f"),
            "V1 Screening Score": st.column_config.NumberColumn(format="%.1f"),
            "Risk-Cost Exposure Index": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    csv_buffer = StringIO()
    employee_view.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download filtered explorer CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="salifort_workforce_explorer_filtered.csv",
        mime="text/csv",
    )
