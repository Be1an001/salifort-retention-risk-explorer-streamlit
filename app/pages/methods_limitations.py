from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_v2_artifact_status, v2_artifacts_available


def render() -> None:
    st.title("Methods & Limitations")
    st.caption("Scope, workflow, and deployment notes for the deployment-focused V1 portfolio app.")

    artifact_status = get_v2_artifact_status()
    available_artifacts = sum(artifact_status.values())
    total_artifacts = len(artifact_status)
    if v2_artifacts_available():
        st.caption(
            f"V2 artifact-ready loaders are enabled. {available_artifacts} of {total_artifacts} "
            "expected V2 artifact files are currently present under `artifacts/v2/`."
        )
    else:
        st.caption(
            "V2 artifact-ready loaders are enabled, but no precomputed V2 artifact files are "
            "present yet. The app is currently running in V1 fallback mode."
        )

    st.subheader("Main Workflow Summary")
    st.markdown(
        "- Clean the checked-in HR dataset and remove duplicate records.\n"
        "- Use notebook-built project artifacts to summarize EDA, validation, threshold tuning, and explainability.\n"
        "- Present the operational model story through a lightweight, deployment-focused Streamlit app.\n"
        "- Use a deterministic V1 screening proxy so the Workforce Explorer stays interactive without in-app retraining."
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

    st.subheader("V1 App Limitations")
    st.markdown(
        "- Advanced model visuals are displayed from existing project artifacts rather than regenerated inside the app.\n"
        "- The app does not retrain models or recalculate SHAP values during runtime.\n"
        "- The Workforce Explorer uses a deterministic V1 screening proxy for interactivity, not the deployed weighted XGBoost probability.\n"
        "- This is an educational portfolio project, not a real company HR system, audit, or production employment workflow."
    )
