from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Methods & Limitations")
    st.caption("Scope, workflow, and deployment notes for the V1 portfolio app.")

    st.subheader("Main Workflow Summary")
    st.markdown(
        "- Clean the checked-in HR dataset and remove duplicate records.\n"
        "- Use notebook-built project artifacts to summarize EDA, validation, threshold tuning, and explainability.\n"
        "- Present the operational model story through a lightweight Streamlit decision app.\n"
        "- Add a deterministic V1 screening proxy so the workforce explorer stays interactive without retraining a model in the app."
    )

    st.subheader("Operational vs Survey-Rich Note")
    st.markdown(
        "This public app is centered on the operational version of the project. "
        "That means it focuses on the fields available in the checked-in HR dataset and on the deployment-friendly workflow, rather than on a survey-rich or retraining-heavy variant."
    )

    st.subheader("Dataset Note")
    st.markdown(
        "The app uses the repo-local `data/hr_capstone_dataset.csv` file. "
        "It is a compact, static dataset that makes the project reproducible for local runs and Streamlit Community Cloud without requiring external data services."
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
        "- The Workforce Explorer uses a lightweight V1 screening proxy for interactivity, not the deployed weighted XGBoost probability.\n"
        "- This is an educational portfolio project and not a real company audit, live HR system, or production employment workflow."
    )
