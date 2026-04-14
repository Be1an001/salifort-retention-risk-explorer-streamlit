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
            "- It turns the broader portfolio workflow into a clear public operational app layer.\n"
            "- It keeps the deployment footprint intentionally lightweight for local runs and Streamlit Community Cloud.\n"
            "- It separates interactive workforce exploration from the heavier original modeling workflow.\n"
            "- It keeps the focus on threshold choice, workforce exposure, and responsible use."
        )

        st.subheader("Interview Explanation")
        st.info(
            "This app is the public operational layer of the Salifort Motors portfolio project. "
            "It is lightweight by design: the interactive explorer uses a separate V1 screening proxy "
            "for deployment simplicity, while the weighted XGBoost model, threshold choice, and SHAP "
            "story come from the documented original workflow and checked-in artifacts."
        )

    with right_col:
        st.image(
            str(figures["14_exec_summary_overview"]),
            caption="Executive summary overview from the original project artifacts.",
            use_container_width=True,
        )
