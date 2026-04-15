from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_runtime_mode_summary, load_v2_employee_scores


def render() -> None:
    st.title("Methods & Limitations")
    st.caption("Scope, workflow, and limitations for this public project app.")

    runtime_summary = get_runtime_mode_summary()
    using_v2_rows = load_v2_employee_scores() is not None
    if runtime_summary["mode"] == "full_v2_artifact_mode":
        st.caption(
            f"Current runtime mode: {runtime_summary['label']}. "
            "The app is using generated files across the supported pages."
        )
    elif runtime_summary["mode"] == "partial_v2_artifact_mode":
        st.caption(
            f"Current runtime mode: {runtime_summary['label']}. "
            f"{runtime_summary['artifacts_present']} of {runtime_summary['artifacts_expected']} "
            "tracked generated files are present, so the app mixes generated sections with built-in fallbacks."
        )
    else:
        st.caption(
            "Current runtime mode: fallback mode. Generated files are not currently available, "
            "so the app is using built-in defaults and reference visuals where needed."
        )

    st.subheader("Main Workflow Summary")
    st.markdown(
        "- Clean the checked-in HR dataset and remove duplicate records.\n"
        "- Use generated project outputs to summarize EDA, validation, threshold tuning, and explainability.\n"
        "- Present the operational model story through a lightweight Streamlit app.\n"
        "- Use generated row-level files when they are available, with a simpler screening fallback for interactive exploration when those files are absent."
    )

    st.subheader("Operational vs Survey-Rich Note")
    st.markdown(
        "This public app is centered on the operational view of the project. "
        "That means it focuses on the fields available in the checked-in HR dataset and on a workflow that is practical to package and share, rather than on a survey-rich or retraining-heavy variant."
    )

    st.subheader("Dataset Note")
    st.markdown(
        "The app uses the repo-local `data/hr_capstone_dataset.csv` file. "
        "It is intentionally kept in the repo to make the project reproducible and to keep Streamlit Community Cloud deployment simple without external data services."
    )

    st.subheader("Why the CSV Remains in the Repo")
    st.markdown(
        "Keeping the CSV in the repository makes the project self-contained. "
        "That improves reproducibility and keeps deployment simple because the app can load everything it needs from version-controlled files."
    )

    st.subheader("Current App Limitations")
    explorer_limit_note = (
        "- The Workforce Explorer uses generated row-level outputs when they are available, but still falls back to a simpler screening score when those files are missing.\n"
        if using_v2_rows
        else "- The Workforce Explorer uses a simpler screening score for interactivity, not the final weighted XGBoost probability.\n"
    )
    st.markdown(
        (
            "- Advanced model visuals are displayed from existing project artifacts rather than regenerated inside the app.\n"
            "- The app does not retrain models or recalculate SHAP values during runtime.\n"
            + explorer_limit_note
            + "- This is an educational project, not a real company HR system, audit, or production employment workflow."
        )
    )

    st.subheader("How Generated Files Fit In")
    st.markdown(
        "- The app can prefer generated files when they are available.\n"
        "- Generated files can replace parts of the static presentation layer for metadata, model comparison, threshold analysis, department exposure, and explainability.\n"
        "- Offline artifact generation remains the source of truth; Streamlit runtime should only load those outputs, not rebuild them."
    )
