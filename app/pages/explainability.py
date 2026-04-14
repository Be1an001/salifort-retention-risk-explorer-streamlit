from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_figure_paths


def render() -> None:
    figures = get_figure_paths()

    st.title("Explainability")
    st.caption(
        "Model interpretation view using existing SHAP artifacts from the original project presentation layer."
    )

    st.image(
        str(figures["16_exec_summary_shap"]),
        caption="Executive SHAP summary view.",
        use_container_width=True,
    )

    figure_cols = st.columns(2, gap="large")
    figure_cols[0].image(
        str(figures["12_shap_summary_plot"]),
        caption="SHAP summary plot.",
        use_container_width=True,
    )
    figure_cols[1].image(
        str(figures["13_shap_dependence_number_project"]),
        caption="SHAP dependence plot for project count.",
        use_container_width=True,
    )

    st.subheader("Top Drivers in Plain Language")
    st.markdown(
        "- Satisfaction level: lower satisfaction is one of the clearest warning signals.\n"
        "- Number of projects: heavier project load increases model concern, especially at the upper end.\n"
        "- Average monthly hours: long sustained work hours reinforce workload pressure.\n"
        "- Tenure: longer time at the company can matter differently depending on promotion history and workload.\n"
        "- Promotion history and salary context: stalled progression and lower pay bands contribute to screening risk."
    )

    st.subheader("How to Interpret These Visuals")
    st.markdown(
        "These SHAP visuals are included to communicate model logic from the broader portfolio workflow. "
        "In this public app, they support the explanation layer while the main framing remains the operational decision-support version focused on screening, threshold choice, and manager review."
    )

    st.warning(
        "SHAP is used here as model interpretation support, not causal HR truth. A feature showing high explanatory importance does not prove that changing that feature will cause retention to improve."
    )
