from __future__ import annotations

import streamlit as st

from app.utils.load_data import get_runtime_mode_summary, load_v2_employee_scores


def render() -> None:
    st.title("Methods & Limitations")
    st.caption("A clear guide to what this portfolio app does, how it runs, and where its limits are.")

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

    st.subheader("How the Project Works")
    st.markdown(
        "The project has two layers: an offline project-building layer and a public Streamlit app layer. "
        "The offline layer prepares data, model outputs, tables, and figures. The Streamlit layer loads those files "
        "and presents them for review."
    )
    st.markdown(
        "- Clean the checked-in HR dataset and remove duplicate records.\n"
        "- Build model outputs, threshold tables, SHAP summaries, and figures outside Streamlit.\n"
        "- Load generated files into the app when they are available.\n"
        "- Keep the web app focused on explanation, exploration, and responsible review."
    )

    st.subheader("What This Public App Focuses On")
    st.markdown(
        "This public app focuses on the fields available in the checked-in HR dataset and on a workflow that is practical to package and share. "
        "It is not a production HR platform, and it does not retrain models while a visitor is using the site."
    )

    st.subheader("Architecture Overview")
    st.markdown(
        "- **Dataset:** `data/hr_capstone_dataset.csv` is checked into the repo so the demo is reproducible.\n"
        "- **Cleaning:** the app loader standardizes column names and removes duplicates before showing metrics.\n"
        "- **Generated artifacts:** files in `artifacts/v2/` hold model metadata, row-level scores, threshold tables, department exposure, and SHAP summaries.\n"
        "- **Static figures:** PNGs in `outputs/figures/` preserve the original visual story for EDA, validation, threshold tuning, and explainability.\n"
        "- **Streamlit runtime:** Streamlit reads local files and renders pages; it does not train models or run background workflows.\n"
        "- **MLOps Lab:** the ninth page adds hosted Online CSV Insight, packaged demo model inference, and committed MLOps Evidence Pack snapshots while keeping local/dev pipeline, MLflow, FastAPI, Docker, Airflow, and CI tooling separate from public app runtime."
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

    st.subheader("Important Limits")
    explorer_limit_note = (
        "- The Workforce Explorer uses generated row-level outputs when they are available, but still falls back to a simpler screening score when those files are missing.\n"
        if using_v2_rows
        else "- The Workforce Explorer uses a simpler screening score for interactivity, not the final weighted XGBoost probability.\n"
    )
    st.markdown(
        (
            "- Model visuals are loaded from existing project artifacts rather than regenerated inside the app.\n"
            "- The app does not retrain models or recalculate SHAP values while it runs.\n"
            + explorer_limit_note
            + "- Online CSV Insight heuristic scores are transparent review-priority signals, not model probabilities.\n"
            + "- Packaged demo model probabilities are MLOps Lab online demo signals, not the public weighted XGBoost threshold `0.29` model story.\n"
            + "- Optional OpenAI briefings use compact aggregate JSON only; raw uploaded rows and identifier-like fields are excluded.\n"
            + "- This is an educational project, not a real company HR system, audit, or production employment workflow."
        )
    )

    st.subheader("How Generated Files Fit In")
    st.markdown(
        "- The app can prefer generated files when they are available.\n"
        "- Generated files can replace parts of the static presentation layer for metadata, model comparison, threshold analysis, department exposure, and explainability.\n"
        "- Offline artifact generation remains the source of truth; Streamlit runtime should only load those outputs, not rebuild them."
    )

    st.subheader("Fallback Logic")
    st.markdown(
        "If row-level generated artifacts are missing, some interactive views use a simpler screening score so the app remains explorable. "
        "That fallback is clearly labeled and should not be confused with the final weighted XGBoost probability."
    )

    st.subheader("Advanced Navigator Concepts")
    st.markdown(
        "- **PACE:** a simple project map: Plan, Analyze, Construct, Execute. It helps organize the project story.\n"
        "- **Retrieval / RAG preparation:** selected docs and registry entries are prepared into traceable chunks. Fixed review questions retrieve those chunks and show citations.\n"
        "- **OpenAI API use:** the API is used for embeddings when building or querying the local retrieval index. The key must come from a local environment variable and is never stored in the repo.\n"
        "- **Answer assembly:** fixed questions can produce structured answers from retrieved evidence, with citations and caveats separated.\n"
        "- **Airflow scaffold:** the repo includes local workflow scaffolding for future orchestration review. Streamlit does not run Airflow jobs.\n"
        "- **Agent shell:** the Navigator includes a controlled plan-preview area. It maps fixed request types to known workflows but does not execute them.\n"
        "- **MLOps Lab:** hosted CSV insight and static evidence are reviewer surfaces; local/dev FastAPI, Docker, MLflow, and Airflow are not required for the app to open."
    )

    st.subheader("Production-Like vs Portfolio-Only")
    st.markdown(
        "- **Production-like:** clear data contracts, offline artifact generation, runtime fallbacks, citations, validation scripts, and responsible-use boundaries.\n"
        "- **Portfolio/demo-only:** no live HR data feed, no production scheduler, no autonomous agent execution, no employee action workflow, and no hidden model retraining.\n"
        "- **Reviewer-only:** advanced retrieval, source previews, audit exports, and readiness panels are included to make the project easier to inspect."
    )
