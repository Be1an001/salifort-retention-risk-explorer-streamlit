from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_figure_paths


def render() -> None:
    figures = get_figure_paths()

    st.title("Manager Action View")
    st.caption("Decision-support framing for prioritization, review, and responsible intervention.")

    top_cols = st.columns(2, gap="large")
    top_cols[0].image(
        str(figures["11_department_exposure_total_normalized"]),
        caption="Department exposure shown as total and normalized views.",
        use_container_width=True,
    )
    top_cols[1].image(
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
