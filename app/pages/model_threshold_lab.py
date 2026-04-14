from __future__ import annotations

import streamlit as st

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback for minimal environments
    px = None

from app.utils.load_data import (
    artifacts_available,
    get_figure_paths,
    get_runtime_mode_label,
    load_preferred_metadata,
    load_v2_confusion_matrix,
    load_v2_pr_curve_points,
    load_v2_threshold_curve,
    load_v2_validation_model_comparison,
)


def _format_percent(value: float) -> str:
    percent_value = value * 100 if value <= 1 else value
    return f"{percent_value:.1f}%"


def render() -> None:
    figures = get_figure_paths()
    metadata = load_preferred_metadata()
    threshold_curve = load_v2_threshold_curve()
    validation_comparison = load_v2_validation_model_comparison()
    confusion_matrix = load_v2_confusion_matrix()
    pr_curve_points = load_v2_pr_curve_points()

    has_core_v2_model_tables = artifacts_available(
        [
            "metadata",
            "threshold_curve",
            "validation_model_comparison",
            "confusion_matrix_at_selected_threshold",
        ]
    )

    st.title("Model & Threshold Lab")
    st.caption("Public operational model view using existing project artifacts rather than in-app retraining.")
    st.caption(f"Runtime mode: {get_runtime_mode_label()}.")
    st.caption(
        "This page is using precomputed V2 model tables where available."
        if has_core_v2_model_tables
        else "This page is using V1 static visuals for any model sections that do not yet have V2 artifacts."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Champion Model", str(metadata["final_model"]))
    metric_cols[1].metric("Threshold", f"{float(metadata['selected_threshold']):.2f}")
    metric_cols[2].metric("Recall", _format_percent(float(metadata["selected_test_recall"])))
    metric_cols[3].metric("Precision", _format_percent(float(metadata["selected_test_precision"])))
    metric_cols[4].metric("Accuracy", _format_percent(float(metadata["selected_test_accuracy"])))

    first_row = st.columns(2, gap="large")
    with first_row[0]:
        st.subheader("Validation Comparison")
        if validation_comparison is not None and not validation_comparison.empty:
            st.dataframe(validation_comparison, use_container_width=True, hide_index=True)
        else:
            st.image(
                str(figures["07_validation_metric_comparison"]),
                caption="Validation metric comparison across candidate models.",
                use_container_width=True,
            )

    with first_row[1]:
        st.subheader("Precision-Recall Context")
        if pr_curve_points is not None and not pr_curve_points.empty and {"model", "recall", "precision"}.issubset(pr_curve_points.columns):
            if px is not None:
                fig = px.line(
                    pr_curve_points,
                    x="recall",
                    y="precision",
                    color="model",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pr_curve_points, use_container_width=True, hide_index=True)
        else:
            st.image(
                str(figures["08_validation_pr_curves"]),
                caption="Precision-recall curves for model comparison.",
                use_container_width=True,
            )

    second_row = st.columns(2, gap="large")
    with second_row[0]:
        st.subheader("Threshold Curve")
        if threshold_curve is not None and not threshold_curve.empty and "threshold" in threshold_curve.columns:
            threshold_view = threshold_curve.copy()
            plot_columns = [
                column
                for column in ["precision", "recall", "f1", "f2", "accuracy"]
                if column in threshold_view.columns
            ]
            if px is not None and plot_columns:
                long_df = threshold_view.melt(
                    id_vars=["threshold"],
                    value_vars=plot_columns,
                    var_name="metric",
                    value_name="value",
                )
                fig = px.line(long_df, x="threshold", y="value", color="metric")
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(threshold_view, use_container_width=True, hide_index=True)
        else:
            st.image(
                str(figures["09_threshold_tuning_xgb_weighted"]),
                caption="Threshold tuning for the weighted XGBoost model.",
                use_container_width=True,
            )

    with second_row[1]:
        st.subheader("Selected Threshold Confusion Matrix")
        if confusion_matrix is not None and not confusion_matrix.empty:
            matrix_row = confusion_matrix.iloc[0]
            matrix_df = None
            if {"tn", "fp", "fn", "tp"}.issubset(confusion_matrix.columns):
                matrix_df = confusion_matrix.loc[:, ["tn", "fp", "fn", "tp"]].rename(
                    columns={
                        "tn": "TN",
                        "fp": "FP",
                        "fn": "FN",
                        "tp": "TP",
                    }
                )
            if matrix_df is not None:
                st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            metric_fields = [field for field in ["precision", "recall", "f1", "f2", "accuracy"] if field in confusion_matrix.columns]
            if metric_fields:
                summary_cols = st.columns(len(metric_fields))
                for column_widget, field in zip(summary_cols, metric_fields):
                    column_widget.metric(field.upper(), _format_percent(float(matrix_row[field])))
        else:
            st.image(
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
