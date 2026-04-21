# Salifort Motors Retention Risk Explorer

Portfolio Streamlit app for reviewing employee retention risk in the Salifort Motors HR dataset.

Live app: https://salifort-retention-risk-explorer.streamlit.app/

## Project Question

How can Salifort Motors spot early retention risk, focus manager attention, and avoid treating a model score as an automated HR decision?

## What This Project Is

This repository is the public app layer for the Salifort Motors retention-risk project. It packages the cleaned dataset, generated model artifacts, and project visuals into a Streamlit demo that is easy for recruiters, hiring managers, and interviewers to explore.

The app is meant for portfolio review and discussion. It is not a production HR platform and should not be used to make employment decisions.

## What the App Shows

- Workforce patterns by department, salary, tenure, workload, and observed attrition.
- Model and threshold trade-offs for a public reference model: weighted XGBoost at threshold `0.29`.
- Explainability support using SHAP outputs.
- Department exposure views for manager review and prioritization.
- Methods, limitations, and responsible-use boundaries.
- Advanced PACE Navigator review tools for citations, retrieval evidence, workflow contracts, and demo readiness.

## Page Guide

- **Overview**: Start here for the project question, headline metrics, and the portfolio summary.
- **PACE Navigator**: Guided project map plus advanced technical review tools.
- **Workforce Explorer**: Filter employee and department patterns interactively.
- **EDA & Patterns**: View the stable analysis figures that support the project story.
- **Model & Threshold Lab**: Compare model performance and threshold trade-offs.
- **Explainability**: See which features most influence the model's risk signal.
- **Manager Action View**: Translate exposure patterns into responsible review priorities.
- **Methods & Limitations**: Understand data, runtime behavior, and project boundaries.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Runtime and Artifacts

The Streamlit app is designed to load existing project files. It does not retrain models or regenerate SHAP values while the app runs.

Generated files live under `artifacts/v2/` when available. If some generated row-level files are missing, selected interactive views fall back to a simpler screening score so the portfolio demo can still be explored. That fallback is not the final model probability.

Offline artifact generation lives in `scripts/build_v2_artifacts.py` and should be run separately from Streamlit.

## Advanced Review Features

The PACE Navigator includes reviewer-oriented tools such as fixed-question retrieval, answer citations, source-preview eligibility, audit exports, orchestration contracts, and plan-preview guardrails.

These features are included to make the project more inspectable. They are controlled review tools, not a chatbot, autonomous agent, production scheduler, or HR decision engine.

Some retrieval-backed reviewer features require a local OpenAI API key supplied through environment variables. No API key or secret should be committed to this repository.

## Key Folders

- `app/`: Streamlit app, pages, loaders, services, and view models.
- `data/`: Checked-in HR dataset used by the app.
- `outputs/figures/`: Stable project figures used in app pages.
- `artifacts/v2/`: Generated model and explanation artifacts consumed by the app.
- `navigator/`: Registries, retrieval packs, readiness contracts, and advanced review metadata.
- `scripts/`: Offline builders and validation scripts.
- `docs/`: Walkthroughs and navigator implementation notes.
