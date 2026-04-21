from __future__ import annotations

import streamlit as st

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback for minimal environments
    px = None

from app.utils.load_data import get_figure_paths, get_runtime_mode_label, load_v2_department_exposure


def render() -> None:
    figures = get_figure_paths()
    department_exposure = load_v2_department_exposure()

    st.title("Manager Action View")
    st.caption("Turn risk patterns into practical, responsible review priorities.")
    st.caption(f"Runtime mode: {get_runtime_mode_label()}.")
    st.markdown(
        "**How to use this page:** review department exposure first, then read the practical priorities and responsible-use note. "
        "The purpose is to focus human attention, not to automate HR action."
    )

    st.subheader("Department Exposure")
    st.caption(
        "This chart and table show where potential retention exposure is concentrated by department. "
        "Compare total exposure with department size and observed attrition."
    )
    if department_exposure is not None and not department_exposure.empty:
        chart_df = department_exposure.sort_values("exposure_index", ascending=True)
        if px is not None and {"department", "exposure_index"}.issubset(chart_df.columns):
            fig = px.bar(
                chart_df,
                x="exposure_index",
                y="department",
                orientation="h",
                color="observed_attrition_rate" if "observed_attrition_rate" in chart_df.columns else None,
                color_continuous_scale="Tealgrn",
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Exposure Index",
                yaxis_title="Department",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Using generated department exposure at threshold {float(department_exposure['selected_threshold'].iloc[0]):.2f} "
            f"for {department_exposure['model_mode'].iloc[0]} mode."
            if {"selected_threshold", "model_mode"}.issubset(department_exposure.columns)
            else "Using generated department exposure data."
        )
        exposure_view = department_exposure.rename(
            columns={
                "department": "Department",
                "headcount": "Headcount",
                "observed_attrition_rate": "Observed Attrition Rate",
                "avg_predicted_attrition": "Avg Predicted Attrition",
                "high_risk_employees": "High-Risk Employees",
                "exposure_index": "Exposure Index",
                "exposure_per_100_employees": "Exposure per 100 Employees",
                "selected_threshold": "Selected Threshold",
                "model_mode": "Model Mode",
            }
        )
        st.dataframe(
            exposure_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Headcount": st.column_config.NumberColumn(format="%d"),
                "Observed Attrition Rate": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Predicted Attrition": st.column_config.NumberColumn(format="%.1f%%"),
                "High-Risk Employees": st.column_config.NumberColumn(format="%d"),
                "Exposure Index": st.column_config.NumberColumn(format="%.2f"),
                "Exposure per 100 Employees": st.column_config.NumberColumn(format="%.2f"),
                "Selected Threshold": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.image(
            str(figures["11_department_exposure_total_normalized"]),
            caption="Department exposure shown as total and normalized views.",
            use_container_width=True,
        )

    st.image(
        str(figures["15_exec_summary_decision_threshold"]),
        caption="Decision-threshold summary for the operational workflow.",
        use_container_width=True,
    )

    st.subheader("How to Compare Departments")
    st.markdown(
        "Total exposure shows where the largest possible retention burden sits. "
        "Normalized exposure helps compare departments more fairly when headcount differs. "
        "Together, they show both scale and intensity."
    )

    st.subheader("Practical Review Priorities")
    st.markdown(
        "- Start with departments that are high on both total and normalized exposure.\n"
        "- Use the threshold view to size review queues instead of treating every flagged employee the same way.\n"
        "- Pair screening with workload, promotion, and team context before deciding whether a retention conversation is appropriate.\n"
        "- Look for repeated workload or career-stall patterns before choosing any action."
    )

    st.subheader("How to Use This Responsibly")
    st.warning(
        "This app is for early-warning screening only. It should support human review, not automate HR decisions. "
        "No employee should be penalized, ranked for employment action, or evaluated in isolation based on this workflow alone."
    )

    st.markdown(
        "Managers and HR partners should treat the screening output as a prompt to investigate workload, development opportunities, team conditions, and local context before acting."
    )
