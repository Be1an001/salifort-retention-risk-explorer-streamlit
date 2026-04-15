from __future__ import annotations

import streamlit as st

from app.utils.load_data import (
    get_figure_paths,
    load_preferred_metadata,
    load_v2_employee_scores,
    load_v2_metadata,
)


def render() -> None:
    figures = get_figure_paths()
    metadata = load_preferred_metadata()
    using_v2_metadata = load_v2_metadata() is not None
    using_v2_rows = load_v2_employee_scores() is not None
    final_model_display = (
        "Weighted XGBoost" if str(metadata["final_model"]).strip().lower() == "weighted xgboost" else str(metadata["final_model"])
    )

    def format_percent(value: float) -> str:
        percent_value = value * 100 if value <= 1 else value
        return f"{percent_value:.1f}%"

    st.title("Salifort Motors Retention Risk Explorer")
    st.caption("Operational HR Analytics Decision App")
    st.caption(
        "Runtime status: using generated project metadata."
        if using_v2_metadata
        else "Runtime status: using built-in project metrics."
    )

    st.markdown(
        "### Project Question\n"
        "How can Salifort Motors identify attrition exposure early enough to support targeted, "
        "responsible retention action without turning the workflow into an automated HR decision engine?"
    )

    top_row = st.columns(3)
    top_row[0].metric("Rows After Cleaning", f"{int(metadata['dataset_rows_clean']):,}")
    top_row[1].metric("Attrition Rate", format_percent(float(metadata["attrition_rate_clean"])))
    top_row[2].metric("Duplicates Removed", f"{int(metadata['duplicates_removed']):,}")

    st.markdown(f"**Final Model:** {final_model_display}")

    bottom_row = st.columns(4)
    bottom_row[0].metric("Decision Threshold", f"{float(metadata['selected_threshold']):.2f}")
    bottom_row[1].metric("Test Recall", format_percent(float(metadata["selected_test_recall"])))
    bottom_row[2].metric("Test Precision", format_percent(float(metadata["selected_test_precision"])))
    bottom_row[3].metric("Test Accuracy", format_percent(float(metadata["selected_test_accuracy"])))

    st.divider()

    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.subheader("Project Summary")
        st.markdown(
            "- It brings the dataset, model outputs, and explanation layer into one clear decision-support app.\n"
            "- It keeps the app simple to run locally and easy to deploy without external services.\n"
            "- It separates offline model building from the app runtime, which keeps the web app stable and readable.\n"
            "- It focuses on threshold choice, department exposure, and responsible review."
        )

        st.subheader("About This App")
        st.info(
            "This app presents the operational side of the Salifort Motors attrition project. "
            "The weighted XGBoost model, threshold choice, and SHAP results come from the documented offline workflow and generated project artifacts. "
            + (
                "The current runtime is using generated row-level artifacts for supported explorer views."
                if using_v2_rows
                else "When generated row-level artifacts are not available, the explorer falls back to a lighter screening view."
            )
        )

    with right_col:
        st.image(
            str(figures["14_exec_summary_overview"]),
            caption="Executive summary overview from the original project artifacts.",
            use_container_width=True,
        )
