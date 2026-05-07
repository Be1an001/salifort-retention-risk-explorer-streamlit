# Salifort Motors Retention Risk Explorer

Portfolio Streamlit app for reviewing employee retention risk in the Salifort Motors HR dataset.

Live app: https://salifort-retention-risk-explorer.streamlit.app/

## Who This Is For

This repo is designed for recruiters, hiring managers, interviewers, and technical reviewers who want to see a complete analytics product rather than only a notebook. It shows how a retention-risk modeling project can be packaged into a readable Streamlit app with clear limitations and responsible-use boundaries.

This is not a production HR platform and should not be used to make employment decisions.

## Business Question

How can Salifort Motors spot early retention risk, focus manager attention, and avoid treating a model score as an automated HR decision?

## What the App Does

- Explores workforce patterns in the cleaned Salifort HR dataset.
- Shows model and threshold trade-offs for the public reference model.
- Uses SHAP outputs to explain why the model flags risk.
- Shows department exposure to support manager review.
- Explains limitations, runtime behavior, and responsible-use boundaries.
- Provides optional advanced reviewer tools for citations, retrieval evidence, source previews, workflow readiness, and plan previews.
- Presents a read-only MLOps Lab page for local/dev pipeline, API, Docker, Airflow, and CI review.
- Supports a hosted Streamlit CSV Insight sandbox for small Salifort-style uploads, heuristic review scoring, and optional aggregate-only AI briefings.
- Supports Streamlit-native packaged demo model scoring in the MLOps Lab without an external FastAPI backend.
- Includes a sanitized MLOps Evidence Pack so online reviewers can inspect local/dev pipeline, serving, orchestration, Docker, MLflow, and CI proof without running the user's machine.

## Architecture Overview

The project has an offline build layer and a Streamlit app layer.

- **Dataset:** `data/hr_capstone_dataset.csv` is checked into the repo so the app is reproducible.
- **Cleaning:** `app/utils/load_data.py` standardizes the dataset and removes duplicates before metrics are shown.
- **Generated artifacts:** `artifacts/v2/` contains model metadata, row-level scores, threshold tables, department exposure, and SHAP summaries when available.
- **Static figures:** `outputs/figures/` contains stable PNG figures used for EDA, validation, threshold, and SHAP pages.
- **Streamlit runtime:** `app/app.py` and `app/pages/` load local files and render the app. Streamlit does not retrain models or regenerate SHAP values during a visitor session.
- **Offline builders:** scripts in `scripts/` can rebuild artifacts or validate advanced Navigator assets outside Streamlit.

For a page-by-page tour, see the [Streamlit app walkthrough](docs/user-guide/streamlit-app-walkthrough.md). For the full docs index, see [docs/README.md](docs/README.md).

An optional local/dev Docker Compose demo for the MLOps Mini-Lab is documented in the [Docker local runbook](docs/mlops-docker-local-runbook.md). It does not change the public Streamlit model truth.

An optional local/dev Airflow DAG scaffold for the MLOps Mini-Lab is documented in the [Airflow local runbook](docs/mlops-airflow-local-runbook.md). It orchestrates lab CLI scripts only and does not run from Streamlit.

GitHub Actions CI checks for the app and MLOps Mini-Lab are documented in the [CI runbook](docs/mlops-ci-runbook.md). CI validates contracts and configuration without deploying or publishing generated artifacts.

For a reviewer-friendly path through the hosted CSV sandbox, local pipeline, FastAPI, Docker, MLflow, Airflow, and CI evidence, see the [MLOps Mini-Lab demo guide](docs/mlops-demo-guide.md).

## Documentation Map

Use these documents when you want more detail than this README:

- [Documentation Guide](docs/README.md): central index for all project docs.
- [Product Requirements Document](docs/product/product-requirements-document.md): business framing, target audiences, scope, and non-goals.
- [Technical Design and Architecture](docs/technical/technical-design-and-architecture.md): runtime layers, artifacts, retrieval design, workflow contracts, and boundaries.
- [Environment Setup and Deployment Guide](docs/deployment/environment-setup-and-deployment-guide.md): local setup, optional API configuration, and deployment steps.
- [User Manual](docs/user-guide/user-manual.md): how to use the app responsibly and what each page is for.
- [HR Quick Start](docs/user-guide/hr-quick-start.md): one-page business guide for HR and department-manager style review.
- [Executive Summary](docs/executive/executive-summary.md): non-technical management summary and governance framing.
- [Formal Documentation Package](docs/formal/salifort-formal-document-package.md): formal docs index, document-control table, screenshot placeholders, and maintenance rules.
- [Streamlit App Walkthrough](docs/user-guide/streamlit-app-walkthrough.md): page-by-page review order and guidance.
- [Navigator Notes](docs/navigator/README.md): advanced reviewer documentation for the PACE Navigator.
- [MLOps Mini-Lab Demo Guide](docs/mlops-demo-guide.md): hosted and local/dev demonstration paths for the MLOps extension.
- [Repository Agent Guidance](AGENTS.md): repo-scoped instructions for Codex and future coding agents.
- [Repo-Scoped Skills](.agents/skills/): focused review workflows for audits, docs refreshes, Streamlit page review, MLOps validation, and responsible AI review.
- [Responsible-Use Eval Cases](evals/README.md): lightweight JSONL cases for AI briefing and safety-boundary review.

## Model and Threshold Truth

The public portfolio story preserves:

- **Public reference model:** weighted XGBoost.
- **Selected threshold:** `0.29`.
- **Runtime approach:** load generated artifacts when available; do not train models inside Streamlit.

If row-level artifacts are missing, selected app views can fall back to a simpler screening score so the demo remains explorable. That fallback is clearly separated from the final model probability.

## Page Guide

The Streamlit sidebar contains nine pages:

- **Overview:** Explains the project question, dataset, model, threshold, and suggested review path. Start here.
- **PACE Navigator:** Gives a guided project map first, then optional advanced review tools for fixed-question answers, citations, retrieval evidence, workflow readiness, and plan preview.
- **Workforce Explorer:** Lets reviewers filter departments, salary bands, tenure bands, and risk flags to inspect workforce slices and department exposure.
- **EDA & Patterns:** Shows stable visual evidence for workload, salary, department, tenure, and project-load patterns.
- **Model & Threshold Lab:** Compares model metrics and explains how the selected threshold changes recall, precision, false positives, and review workload.
- **Explainability:** Uses SHAP outputs to explain which features influence the model signal, while keeping causal claims off-limits.
- **Manager Action View:** Turns exposure patterns into practical review priorities and responsible-use guidance.
- **Methods & Limitations:** Explains the architecture, artifacts, fallback logic, PACE, retrieval, Airflow scaffold, agent shell, and production boundaries.
- **MLOps Lab:** Shows the hosted Online CSV Insight sandbox, packaged demo model inference, MLOps Evidence Pack, and local/dev pipeline, MLflow, FastAPI, Docker Compose, Airflow DAG, and CI checks. Hosted CSV Insight does not require FastAPI, Docker, MLflow, Airflow, Render, or an external API. Optional AI briefings use compact aggregate summaries rather than raw CSV rows.

## Suggested Reading Order

1. Overview
2. Workforce Explorer
3. EDA & Patterns
4. Model & Threshold Lab
5. Explainability
6. Manager Action View
7. Methods & Limitations
8. PACE Navigator if you want the advanced governed reviewer layer
9. MLOps Lab if you want hosted CSV insight, packaged demo inference, and local/dev MLOps evidence

## What PACE Means Here

PACE means Plan, Analyze, Construct, and Execute. In this repo it is a project map that helps organize the work:

- **Plan:** frame the business question and responsible-use boundaries.
- **Analyze:** explore workforce patterns and data signals.
- **Construct:** build the model story, artifacts, threshold view, and explanations.
- **Execute:** present review-ready outputs and manager-facing decision support.

## Retrieval, RAG, and Advanced Review Tools

The PACE Navigator includes advanced review surfaces built from prepared project files.

- **Retrieval pack:** selected project metadata and documentation are converted into traceable chunks.
- **Embedding index:** those chunks can be embedded locally through the OpenAI API when the user supplies their own key.
- **Answer viewer:** fixed review questions plus retrieval depth can retrieve relevant chunks and assemble structured, citation-backed answers.
- **Source preview:** small eligible source files can be previewed safely from governed paths.
- **Audit exports:** reviewer summaries can be exported as markdown, text, or JSON.

This is retrieval-backed review support, not a chatbot and not open-ended answer generation.

## Airflow and Agent Shell Boundaries

The repo includes workflow contracts and an Airflow-ready scaffold so reviewers can inspect how future offline jobs could be organized. Streamlit does not run Airflow jobs.

The agent shell is a controlled plan-preview surface. It maps fixed request types to known workflows and blockers. It does not execute workflows, trigger background jobs, or act autonomously.

## Local Run

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Some advanced retrieval-backed reviewer features require a local OpenAI API key supplied through environment variables such as `RAG_STREAMLIT_OPENAI_API_KEY` or `OPENAI_API_KEY`. No API key or secret should be committed to this repository.

The hosted MLOps Lab can also use these optional secrets:

- `OPENAI_API_KEY` for aggregate-only AI briefings.
- `OPENAI_SUMMARY_MODEL`, defaulting to `gpt-5.4-mini`.

The hosted MLOps Lab no longer requires `SALIFORT_API_URL` or `SALIFORT_API_TOKEN`. Local/dev FastAPI remains available for technical review, but it is not required for the hosted CSV Insight sandbox.

The hosted packaged model scoring path uses `scikit-learn`, `xgboost`, and `joblib` in `requirements.txt`. Local/dev MLOps tooling such as MLflow, FastAPI, and pytest remains in `requirements-mlops.txt`.

Optional local/dev MLOps path:

```bash
pip install -r requirements-mlops.txt
python scripts/mlops_run_pipeline.py
python scripts/export_mlops_evidence_pack.py
python scripts/export_streamlit_model_artifact.py
python -m uvicorn api.main:app --reload
python scripts/validate_mlops_airflow_dag.py
```

Optional Docker demo:

```bash
docker compose config
docker compose up api
docker compose up streamlit
docker compose --profile mlflow up mlflow
```

See the [Docker local runbook](docs/mlops-docker-local-runbook.md) for API, Streamlit, and MLflow service details.

Optional Airflow scaffold validation:

```bash
python scripts/validate_mlops_airflow_dag.py
```

See the [Airflow local runbook](docs/mlops-airflow-local-runbook.md) for DAG setup notes.

CI details are available in the [CI runbook](docs/mlops-ci-runbook.md).

## Key Folders

- `app/`: Streamlit app, pages, loaders, services, and view models.
- `data/`: checked-in HR dataset used by the app.
- `outputs/figures/`: stable project figures used in app pages.
- `artifacts/v2/`: generated model and explanation artifacts consumed by the app.
- `navigator/`: registries, retrieval packs, readiness contracts, and advanced review metadata.
- `scripts/`: offline builders and validation scripts.
- `docs/`: PRD, TDD, deployment guide, user guides, formal package, runbooks, evidence snapshots, and Navigator implementation notes.
- `.agents/skills/`: repo-scoped Codex skills for repeatable review workflows.
- `evals/`: lightweight responsible-use and AI briefing behavior cases.

## Portfolio vs Production Boundary

Production-like ideas in this repo include clear runtime boundaries, generated artifact contracts, validation scripts, citation-backed review, and responsible-use language.

Portfolio/demo-only boundaries include no live HR data feed, no production scheduler, no autonomous agent execution, no employee action workflow, and no hidden model retraining.
