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


FEATURE_EXPLANATIONS = {
    "tenure_x_projects": "Tenure x projects: sustained workload over time is a leading driver in the operational model.",
    "last_evaluation": "Last evaluation: stronger evaluation signals can align with elevated review attention when paired with workload patterns.",
    "average_monthly_hours": "Average monthly hours: longer sustained hours reinforce workload pressure.",
    "number_project": "Number of projects: heavier project load increases model concern, especially at the upper end.",
    "project_intensity": "Project intensity: the hours-to-project mix helps the model distinguish concentrated workload pressure from normal variation.",
    "career_stall_flag": "Career stall flag: longer tenure without promotion contributes to screening concern.",
    "tenure": "Tenure: time at the company matters differently depending on project load and progression history.",
    "salary_level": "Salary level: lower pay context can contribute to elevated screening risk in the operational workflow.",
    "work_accident": "Work accident: the model uses accident history as part of the broader employment-pattern context, not as a standalone causal claim.",
    "undervalued_flag": "Undervalued flag: high effort with limited advancement remains one of the practical warning patterns.",
    "overworked": "Overworked flag: long hours above the operational workload cutoff reinforce the screening signal.",
    "promotion_last_5years": "Promotion history: stalled progression remains part of the model's interpretation layer.",
}


def _humanize_feature_name(feature: str) -> str:
    if feature.startswith("department_"):
        department_name = feature.replace("department_", "").replace("_", " ")
        return f"Department ({department_name})"
    return feature.replace("_", " ").title()


def _humanize_model_name(model_name: str) -> str:
    mapping = {
        "xgb_weighted": "Weighted XGBoost",
        "weighted XGBoost": "Weighted XGBoost",
    }
    return mapping.get(model_name, model_name.replace("_", " ").title())


def render() -> None:
    figures = get_figure_paths()
    shap_importance = load_v2_shap_importance()
    employee_shap_sample = load_v2_employee_shap_sample()

    st.title("Explainability")
    st.caption(
        "Model interpretation view using generated SHAP outputs and selected project visuals."
    )
    st.caption(f"Runtime mode: {get_runtime_mode_label()}.")

    if shap_importance is not None and not shap_importance.empty:
        model_name = shap_importance["model_name"].iloc[0] if "model_name" in shap_importance.columns else "selected model"
        model_mode = shap_importance["model_mode"].iloc[0] if "model_mode" in shap_importance.columns else "current mode"
        st.caption(
            f"Using generated SHAP importance data for {_humanize_model_name(str(model_name))} in {model_mode} mode."
        )
        shap_view = shap_importance.sort_values("rank").copy()
        display_view = shap_view.copy()
        if "feature" in display_view.columns:
            display_view["feature"] = display_view["feature"].map(_humanize_feature_name)
        if px is not None and {"feature", "mean_abs_shap"}.issubset(shap_view.columns):
            chart_df = shap_view.sort_values("mean_abs_shap", ascending=True)
            chart_df = chart_df.assign(feature_label=chart_df["feature"].map(_humanize_feature_name))
            fig = px.bar(
                chart_df,
                x="mean_abs_shap",
                y="feature_label",
                orientation="h",
                color="rank" if "rank" in chart_df.columns else None,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Mean |SHAP|")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            display_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "feature": st.column_config.TextColumn("Feature"),
                "mean_abs_shap": st.column_config.NumberColumn("Mean |SHAP|", format="%.3f"),
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
            },
        )
    else:
        st.caption("Using the reference SHAP visuals because no generated SHAP table is present.")
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
            "This can support a future employee-level drilldown."
        )

    st.subheader("Top Drivers in Plain Language")
    if shap_importance is not None and not shap_importance.empty:
        top_features = shap_importance.sort_values("rank").head(6)["feature"].tolist()
        bullet_lines = [
            f"- {_humanize_feature_name(feature)}: {FEATURE_EXPLANATIONS.get(feature, 'This feature is one of the stronger contributors in the current operational explanation layer.')}"
            for feature in top_features
        ]
        st.markdown("\n".join(bullet_lines))
        st.caption("The list above follows the current SHAP ranking from the generated artifact.")
    else:
        st.markdown(
            "- Satisfaction level: lower satisfaction is one of the clearest warning signals.\n"
            "- Number of projects: heavier project load increases model concern, especially at the upper end.\n"
            "- Average monthly hours: long sustained work hours reinforce workload pressure.\n"
            "- Tenure: longer time at the company can matter differently depending on promotion history and workload.\n"
            "- Promotion history and salary context: stalled progression and lower pay bands contribute to screening risk."
        )

    st.subheader("How to Interpret These Visuals")
    st.markdown(
        "These SHAP visuals and tables help explain how the model is making distinctions in the project workflow. "
        "In this app, they support the explanation layer while the main focus remains screening, threshold choice, and manager review."
    )

    st.warning(
        "SHAP is used here as model interpretation support, not causal HR truth. A feature showing high explanatory importance does not prove that changing that feature will cause retention to improve."
    )
