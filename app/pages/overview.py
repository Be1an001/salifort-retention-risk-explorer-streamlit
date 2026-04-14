from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_figure_paths


def render() -> None:
    figures = get_figure_paths()

    st.title("Salifort Motors Retention Risk Explorer")
    st.caption("Operational HR Analytics Decision App")

    st.markdown(
        "### Project Question\n"
        "How can Salifort Motors identify attrition exposure early enough to support targeted, "
        "responsible retention action without turning the workflow into an automated HR decision engine?"
    )

    top_row = st.columns(4)
    top_row[0].metric("Rows After Cleaning", "11,991")
    top_row[1].metric("Attrition Rate", "16.6%")
    top_row[2].metric("Duplicates Removed", "3,008")
    top_row[3].metric("Final Model", "Weighted XGBoost")

    bottom_row = st.columns(4)
    bottom_row[0].metric("Decision Threshold", "0.29")
    bottom_row[1].metric("Test Recall", "93.7%")
    bottom_row[2].metric("Test Precision", "81.8%")
    bottom_row[3].metric("Test Accuracy", "95.5%")

    st.divider()

    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.subheader("Why this project is strong")
        st.markdown(
            "- It translates a full notebook-style portfolio project into a business-facing decision app.\n"
            "- It keeps the operational story focused on threshold choice, workforce exposure, and responsible use.\n"
            "- It shows modeling performance alongside practical manager action guidance instead of stopping at metrics.\n"
            "- It keeps the deployment footprint lightweight enough for local runs and Streamlit Community Cloud."
        )

        st.subheader("Interview Explanation")
        st.info(
            "This app is the public operational layer of the Salifort Motors retention project. "
            "The final trained model is a weighted XGBoost classifier tuned around a 0.29 decision "
            "threshold, while the interactive explorer uses a separate V1 screening proxy so the app "
            "can remain lightweight, deterministic, and easy to deploy."
        )

    with right_col:
        st.image(
            str(figures["14_exec_summary_overview"]),
            caption="Executive summary overview from the original project artifacts.",
            use_container_width=True,
        )
