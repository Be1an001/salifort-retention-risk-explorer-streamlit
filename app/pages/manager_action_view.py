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
    st.caption("Decision-support framing for prioritization, review, and responsible intervention.")
    st.caption(f"Runtime mode: {get_runtime_mode_label()}.")

    top_cols = st.columns(2, gap="large")
    with top_cols[0]:
        st.subheader("Department Exposure")
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
            st.dataframe(department_exposure, use_container_width=True, hide_index=True)
        else:
            st.image(
                str(figures["11_department_exposure_total_normalized"]),
                caption="Department exposure shown as total and normalized views.",
                use_container_width=True,
            )

    with top_cols[1]:
        st.image(
            str(figures["15_exec_summary_decision_threshold"]),
            caption="Decision-threshold summary for the operational workflow.",
            use_container_width=True,
        )

    st.subheader("Total vs Normalized Exposure")
    st.markdown(
        "Total exposure highlights where the absolute volume of possible attrition burden is concentrated. "
        "Normalized exposure helps compare departments more fairly when headcount differs. "
        "Together, they help answer two different questions: where the biggest operational burden sits, and where the highest intensity of concern appears."
    )

    st.subheader("Practical HR Action Levers")
    st.markdown(
        "- Prioritize manager review in departments that are high on both total and normalized exposure.\n"
        "- Use the threshold view to size review queues deliberately rather than treating every flagged employee the same way.\n"
        "- Pair screening with workload, promotion, and team context before deciding whether a retention conversation is appropriate.\n"
        "- Track whether intervention attention is landing on teams with repeated workload or career-stall patterns."
    )

    st.subheader("How to Use This Responsibly")
    st.warning(
        "This app is for early-warning screening only. It should support human review, not automate HR decisions. "
        "No employee should be penalized, ranked for employment action, or evaluated in isolation based on this workflow alone."
    )

    st.markdown(
        "Managers and HR partners should treat the screening output as a prompt to investigate workload, development opportunities, team conditions, and local context before acting."
    )
