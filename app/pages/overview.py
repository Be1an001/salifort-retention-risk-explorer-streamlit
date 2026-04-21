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
    st.caption("Portfolio Streamlit demo for responsible employee retention risk review.")
    st.caption(
        "Runtime status: using generated project metadata."
        if using_v2_metadata
        else "Runtime status: using built-in project metrics."
    )

    st.markdown(
        "### Project Question\n"
        "How can Salifort Motors spot early retention risk, focus manager attention, "
        "and avoid treating a model score as an automated HR decision?"
    )
    st.info(
        "This app is a guided portfolio demo. It shows how HR data, generated model outputs, "
        "and clear review boundaries can be packaged into an understandable analytics product."
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
        st.subheader("What This App Is")
        st.markdown(
            "- A Streamlit portfolio app for exploring employee retention risk.\n"
            "- A guided review of workforce patterns, model threshold trade-offs, explainability, and manager-facing priorities.\n"
            "- A demo product that supports human review and discussion, not automated employment decisions."
        )

        st.subheader("What Data and Model Are Used")
        st.markdown(
            f"- **Data:** checked-in Salifort HR dataset after cleaning and duplicate removal.\n"
            f"- **Public reference model:** {final_model_display}.\n"
            f"- **Selected threshold:** {float(metadata['selected_threshold']):.2f}, used to size review queues.\n"
            "- **Artifacts:** generated tables and figures are loaded by the app; model training happens offline."
        )

        st.subheader("What This App Does Not Do")
        st.markdown(
            "- It does not retrain models while a visitor uses the site.\n"
            "- It does not make HR decisions or rank people for employment action.\n"
            "- It does not treat fallback screening scores as final model probabilities."
        )

    with right_col:
        st.image(
            str(figures["14_exec_summary_overview"]),
            caption="Portfolio summary figure from the generated project visuals.",
            use_container_width=True,
        )
        st.caption(
            "If you see a chart or table in this app, read it as review support: useful for questions and prioritization, "
            "not as a standalone decision."
        )

    st.subheader("Where Retrieval and Advanced Review Fit In")
    st.markdown(
        "Most pages load checked-in data, generated artifacts, and static figures. "
        "The PACE Navigator adds an optional reviewer layer: fixed questions retrieve prepared project evidence, "
        "assemble citation-backed answers, and show source traces. The OpenAI API is used only for embeddings when a local key is provided; "
        "the app does not store keys, run a chatbot, or execute jobs from Streamlit."
    )

    st.subheader("Suggested Review Path")
    st.markdown(
        "1. **Overview:** understand the question, data, model, and threshold.\n"
        "2. **Workforce Explorer:** filter departments and employee-level patterns.\n"
        "3. **EDA & Patterns:** review the visual evidence behind the project story.\n"
        "4. **Model & Threshold Lab:** inspect model and threshold trade-offs.\n"
        "5. **Explainability:** see why the model flags risk.\n"
        "6. **Manager Action View:** translate exposure patterns into responsible review priorities.\n"
        "7. **Methods & Limitations:** check architecture, assumptions, and boundaries.\n"
        "8. **PACE Navigator:** optional advanced reviewer tools for citations, retrieval, and workflow readiness."
    )

    st.subheader("What Makes This Portfolio-Ready")
    st.markdown(
        "- It shows end-to-end product thinking, not just a notebook result.\n"
        "- It separates data loading, generated artifacts, app presentation, and advanced review tooling.\n"
        "- It keeps responsible-use language visible throughout the workflow.\n"
        + (
            "- Current status: generated row-level artifacts are available for supported explorer views."
            if using_v2_rows
            else "- Current status: generated row-level artifacts are unavailable, so supported views use the documented fallback."
        )
    )
