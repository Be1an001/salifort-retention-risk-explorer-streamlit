from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_runtime_mode_summary, load_v2_employee_scores


def render() -> None:
    st.title("Methods & Limitations")
    st.caption("Scope, workflow, and deployment notes for the deployment-focused portfolio app.")

    runtime_summary = get_runtime_mode_summary()
    using_v2_rows = load_v2_employee_scores() is not None
    if runtime_summary["mode"] == "full_v2_artifact_mode":
        st.caption(
            f"Current runtime mode: {runtime_summary['label']}. "
            "The app can use the current precomputed V2 artifact set across supported pages."
        )
    elif runtime_summary["mode"] == "partial_v2_artifact_mode":
        st.caption(
            f"Current runtime mode: {runtime_summary['label']}. "
            f"{runtime_summary['artifacts_present']} of {runtime_summary['artifacts_expected']} "
            "tracked V2 artifacts are present, so the app mixes artifact-backed sections with V1 fallbacks."
        )
    else:
        st.caption(
            "Current runtime mode: V1 fallback mode. V2 artifact-aware loaders are enabled, "
            "but no precomputed V2 artifact files are currently present."
        )

    st.subheader("Main Workflow Summary")
    st.markdown(
        "- Clean the checked-in HR dataset and remove duplicate records.\n"
        "- Use notebook-built project artifacts to summarize EDA, validation, threshold tuning, and explainability.\n"
        "- Present the operational model story through a lightweight, deployment-focused Streamlit app.\n"
        "- Use precomputed V2 artifacts when they are available, with a deterministic V1 screening proxy fallback for interactive exploration when row-level artifacts are absent."
    )

    st.subheader("Operational vs Survey-Rich Note")
    st.markdown(
        "This public app is centered on the operational version of the project. "
        "That means it focuses on the fields available in the checked-in HR dataset and on the deployment-friendly workflow, rather than on a survey-rich or retraining-heavy variant."
    )

    st.subheader("Dataset Note")
    st.markdown(
        "The app uses the repo-local `data/hr_capstone_dataset.csv` file. "
        "It is intentionally kept in the repo to make the project reproducible and to keep Streamlit Community Cloud deployment simple without external data services."
    )

    st.subheader("Why the CSV Remains in the Repo")
    st.markdown(
        "Keeping the CSV in the repository makes the portfolio project self-contained. "
        "That improves reproducibility, removes infrastructure friction for reviewers, and keeps deployment simple because the app can load everything it needs from version-controlled files."
    )

    st.subheader("Current App Limitations")
    explorer_limit_note = (
        "- The Workforce Explorer is currently using precomputed V2 row-level outputs where they are available, but still falls back to the deterministic V1 screening proxy if those artifacts are missing.\n"
        if using_v2_rows
        else "- The Workforce Explorer uses a deterministic V1 screening proxy for interactivity, not the deployed weighted XGBoost probability.\n"
    )
    st.markdown(
        (
            "- Advanced model visuals are displayed from existing project artifacts rather than regenerated inside the app.\n"
            "- The app does not retrain models or recalculate SHAP values during runtime.\n"
            + explorer_limit_note
            + "- This is an educational portfolio project, not a real company HR system, audit, or production employment workflow."
        )
    )

    st.subheader("How V2 Artifact Support Changes This")
    st.markdown(
        "- The app is now artifact-aware and can prefer precomputed V2 files when they are available.\n"
        "- Future V2 artifacts can replace parts of the current static presentation layer for metadata, model comparison, threshold analysis, department exposure, and explainability.\n"
        "- Offline artifact generation remains the source of truth; Streamlit runtime should only load those outputs, not rebuild them."
    )
