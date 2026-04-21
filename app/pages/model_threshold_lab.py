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


def _pretty_model_name(value: str) -> str:
    mapping = {
        "xgb_weighted": "Weighted XGBoost",
        "weighted xgboost": "Weighted XGBoost",
        "weighted XGBoost": "Weighted XGBoost",
        "rf_balanced": "Balanced Random Forest",
        "log_reg_balanced": "Balanced Logistic Regression",
        "log_reg_smote": "SMOTE Logistic Regression",
    }
    return mapping.get(value, value)


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
    st.caption("Compare candidate models and see how the selected threshold changes the review trade-off.")
    st.caption(f"Runtime mode: {get_runtime_mode_label()}.")
    st.caption(
        "This page is using generated model tables where they are available."
        if has_core_v2_model_tables
        else "This page is using reference visuals for sections that do not yet have generated tables."
    )

    st.markdown(f"**Champion Model:** {_pretty_model_name(str(metadata['final_model']))}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Threshold", f"{float(metadata['selected_threshold']):.2f}")
    metric_cols[1].metric("Recall", _format_percent(float(metadata["selected_test_recall"])))
    metric_cols[2].metric("Precision", _format_percent(float(metadata["selected_test_precision"])))
    metric_cols[3].metric("Accuracy", _format_percent(float(metadata["selected_test_accuracy"])))

    st.subheader("Model Comparison")
    if validation_comparison is not None and not validation_comparison.empty:
        comparison_view = validation_comparison.rename(
            columns={
                "model": "Model",
                "threshold_0_5_recall": "Recall @ 0.50",
                "threshold_0_5_f1": "F1 @ 0.50",
                "threshold_0_5_pr_auc": "PR AUC @ 0.50",
                "best_threshold": "Best Threshold",
                "best_recall": "Best Recall",
                "best_precision": "Best Precision",
                "best_f1": "Best F1",
                "best_f2": "Best F2",
                "best_roc_auc": "Best ROC AUC",
                "best_pr_auc": "Best PR AUC",
                "best_cost": "Best Cost",
                "model_mode": "Model Mode",
            }
        ).copy()
        comparison_view["Model"] = comparison_view["Model"].map(_pretty_model_name)
        st.dataframe(
            comparison_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Recall @ 0.50": st.column_config.NumberColumn(format="%.1f%%"),
                "F1 @ 0.50": st.column_config.NumberColumn(format="%.3f"),
                "PR AUC @ 0.50": st.column_config.NumberColumn(format="%.3f"),
                "Best Threshold": st.column_config.NumberColumn(format="%.2f"),
                "Best Recall": st.column_config.NumberColumn(format="%.1f%%"),
                "Best Precision": st.column_config.NumberColumn(format="%.1f%%"),
                "Best F1": st.column_config.NumberColumn(format="%.3f"),
                "Best F2": st.column_config.NumberColumn(format="%.3f"),
                "Best ROC AUC": st.column_config.NumberColumn(format="%.3f"),
                "Best PR AUC": st.column_config.NumberColumn(format="%.3f"),
                "Best Cost": st.column_config.NumberColumn(format="%.0f"),
            },
        )
    else:
        st.image(
            str(figures["07_validation_metric_comparison"]),
            caption="Validation metric comparison across candidate models.",
            use_container_width=True,
        )

    st.subheader("Precision and Recall")
    if pr_curve_points is not None and not pr_curve_points.empty and {"model", "recall", "precision"}.issubset(pr_curve_points.columns):
        chart_points = pr_curve_points.copy()
        chart_points["model"] = chart_points["model"].map(_pretty_model_name)
        if px is not None:
            fig = px.line(
                chart_points,
                x="recall",
                y="precision",
                color="model",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show underlying PR curve points"):
            st.dataframe(
                chart_points,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "recall": st.column_config.NumberColumn("Recall", format="%.1f%%"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
                },
            )
    else:
        st.image(
            str(figures["08_validation_pr_curves"]),
            caption="Precision-recall curves for model comparison.",
            use_container_width=True,
        )

    st.subheader("Threshold Trade-Off Curve")
    if threshold_curve is not None and not threshold_curve.empty and "threshold" in threshold_curve.columns:
        threshold_view = threshold_curve.copy()
        selected_threshold = float(metadata["selected_threshold"])
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
            fig.add_vline(x=selected_threshold, line_dash="dash", line_color="#C0392B")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
        selected_rows = threshold_view[threshold_view["threshold"].round(2).eq(round(selected_threshold, 2))]
        if not selected_rows.empty:
            st.caption(f"Selected operating threshold: {selected_threshold:.2f}")
            st.dataframe(
                selected_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.1f%%"),
                    "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                    "f2": st.column_config.NumberColumn("F2", format="%.3f"),
                    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.1f%%"),
                    "cost": st.column_config.NumberColumn("Cost", format="%.0f"),
                    "flagged_count": st.column_config.NumberColumn("Flagged Count", format="%d"),
                    "flagged_rate": st.column_config.NumberColumn("Flagged Rate", format="%.1f%%"),
                },
            )
        with st.expander("Show full threshold curve table"):
            st.dataframe(
                threshold_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.1f%%"),
                    "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                    "f2": st.column_config.NumberColumn("F2", format="%.3f"),
                    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.1f%%"),
                    "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.3f"),
                    "pr_auc": st.column_config.NumberColumn("PR AUC", format="%.3f"),
                    "cost": st.column_config.NumberColumn("Cost", format="%.0f"),
                    "flagged_count": st.column_config.NumberColumn("Flagged Count", format="%d"),
                    "flagged_rate": st.column_config.NumberColumn("Flagged Rate", format="%.1f%%"),
                },
            )
    else:
        st.image(
            str(figures["09_threshold_tuning_xgb_weighted"]),
            caption="Threshold tuning for the weighted XGBoost model.",
            use_container_width=True,
        )

    st.subheader("Selected Threshold Confusion Matrix")
    if confusion_matrix is not None and not confusion_matrix.empty:
        matrix_row = confusion_matrix.iloc[0]
        matrix_df = None
        if {"tn", "fp", "fn", "tp"}.issubset(confusion_matrix.columns):
            matrix_df = confusion_matrix.loc[:, ["tn", "fp", "fn", "tp"]].rename(
                columns={
                    "tn": "True Negatives",
                    "fp": "False Positives",
                    "fn": "False Negatives",
                    "tp": "True Positives",
                }
            )
        if matrix_df is not None:
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
        summary_cols_top = st.columns(3)
        metric_mappings_top = [
            ("Recall", "recall"),
            ("Precision", "precision"),
            ("Accuracy", "accuracy"),
        ]
        for column_widget, (label, field) in zip(summary_cols_top, metric_mappings_top):
            if field in confusion_matrix.columns:
                column_widget.metric(label, _format_percent(float(matrix_row[field])))
        summary_cols_bottom = st.columns(2)
        metric_mappings_bottom = [("F1", "f1"), ("F2", "f2")]
        for column_widget, (label, field) in zip(summary_cols_bottom, metric_mappings_bottom):
            if field in confusion_matrix.columns:
                column_widget.metric(label, f"{float(matrix_row[field]):.3f}")
    else:
        st.image(
            str(figures["10_champion_confusion_matrix"]),
            caption="Champion model confusion matrix at the chosen operating threshold.",
            use_container_width=True,
        )

    st.subheader("Why Precision and Recall Matter")
    st.markdown(
        "Attrition is the smaller outcome in this dataset, so accuracy alone can be misleading. "
        "Precision and recall make the trade-off clearer: how many likely leavers are found, and how many flagged employees need review."
    )

    st.subheader("Why the Threshold Matters")
    st.markdown(
        "The threshold controls how many employees are flagged for review. "
        "A lower threshold catches more possible risk but also creates more false positives and more manager follow-up work. "
        "That choice is a business-review decision, not just a model-training detail."
    )

    st.subheader("How This Supports Review")
    st.info(
        "This page shows how the model could support responsible review in practice: compare trade-offs, "
        "choose a threshold intentionally, and treat the output as an early-warning signal rather than an automated verdict."
    )
