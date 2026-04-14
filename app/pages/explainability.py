from __future__ import annotations

import streamlit as st

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback for minimal environments
    px = None

from app.utils.load_data import (
    get_figure_paths,
    get_runtime_mode_label,
    load_v2_employee_shap_sample,
    load_v2_shap_importance,
)


def render() -> None:
    figures = get_figure_paths()
    shap_importance = load_v2_shap_importance()
    employee_shap_sample = load_v2_employee_shap_sample()

    st.title("Explainability")
    st.caption(
        "Model interpretation view using existing SHAP artifacts from the original project presentation layer."
    )
    st.caption(f"Runtime mode: {get_runtime_mode_label()}.")

    if shap_importance is not None and not shap_importance.empty:
        st.caption("Using precomputed V2 SHAP importance data where available.")
        shap_view = shap_importance.sort_values("rank").copy()
        if px is not None and {"feature", "mean_abs_shap"}.issubset(shap_view.columns):
            chart_df = shap_view.sort_values("mean_abs_shap", ascending=True)
            fig = px.bar(
                chart_df,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                color="rank" if "rank" in chart_df.columns else None,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Mean |SHAP|")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(shap_view, use_container_width=True, hide_index=True)
    else:
        st.caption("Using the V1 SHAP presentation visuals because no V2 SHAP data artifact is present.")
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

    if employee_shap_sample is not None and not employee_shap_sample.empty:
        st.caption(
            f"Optional row-level SHAP sample detected ({len(employee_shap_sample):,} rows). "
            "This is ready for a future deeper employee-level drilldown."
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
        "These SHAP visuals or tables are included to communicate model logic from the broader portfolio workflow. "
        "In this public app, they support the explanation layer while the main framing remains the operational decision-support version focused on screening, threshold choice, and manager review."
    )

    st.warning(
        "SHAP is used here as model interpretation support, not causal HR truth. A feature showing high explanatory importance does not prove that changing that feature will cause retention to improve."
    )
