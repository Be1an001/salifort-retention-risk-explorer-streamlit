from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_figure_paths


def render() -> None:
    figures = get_figure_paths()

    st.title("Model & Threshold Lab")
    st.caption("Public operational model view using existing project artifacts rather than in-app retraining.")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Champion Model", "Weighted XGBoost")
    metric_cols[1].metric("Threshold", "0.29")
    metric_cols[2].metric("Recall", "93.7%")
    metric_cols[3].metric("Precision", "81.8%")
    metric_cols[4].metric("Accuracy", "95.5%")

    first_row = st.columns(2, gap="large")
    first_row[0].image(
        str(figures["07_validation_metric_comparison"]),
        caption="Validation metric comparison across candidate models.",
        use_container_width=True,
    )
    first_row[1].image(
        str(figures["08_validation_pr_curves"]),
        caption="Precision-recall curves for model comparison.",
        use_container_width=True,
    )

    second_row = st.columns(2, gap="large")
    second_row[0].image(
        str(figures["09_threshold_tuning_xgb_weighted"]),
        caption="Threshold tuning for the weighted XGBoost model.",
        use_container_width=True,
    )
    second_row[1].image(
        str(figures["10_champion_confusion_matrix"]),
        caption="Champion model confusion matrix at the chosen operating threshold.",
        use_container_width=True,
    )

    st.subheader("Why PR-Based Evaluation Matters")
    st.markdown(
        "Attrition is the minority outcome in this dataset, so precision-recall analysis is more informative than accuracy alone. "
        "A model can look strong on overall accuracy while still missing many of the employees the business most wants to identify early."
    )

    st.subheader("Why Threshold Choice Is a Business Decision")
    st.markdown(
        "The threshold determines how aggressively the organization wants to screen for possible attrition. "
        "Lower thresholds catch more of the at-risk population but increase false positives, which means more manager review effort. "
        "That trade-off belongs in the operating design, not just inside model training."
    )

    st.subheader("Why the Operational Version Is the Main Public Version")
    st.info(
        "The portfolio app emphasizes the operational version because it best communicates how a model supports decision-making in practice: "
        "compare trade-offs, choose a threshold intentionally, and hand managers an early-warning view instead of an automated verdict."
    )
