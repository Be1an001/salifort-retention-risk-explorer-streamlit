# Environment Setup and Deployment Guide

## Document Purpose

This guide explains how to set up, run, and deploy the **Salifort Motors Retention Risk Explorer**.

It is written for developers and technical reviewers who want to:

- run the app locally
- understand the minimum runtime dependencies
- configure optional retrieval-backed reviewer features
- deploy the app to Streamlit Community Cloud

## Project Runtime Summary

This repo is a Streamlit portfolio app. The main app entry point is:

- `app/app.py`

The runtime is designed to read:

- the checked-in dataset in `data/`
- generated artifacts in `artifacts/v2/`
- static figures in `outputs/figures/`
- advanced reviewer metadata in `navigator/`

The app does not retrain models or rebuild artifacts during a visitor session.

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd salifort-retention-risk-explorer-streamlit
```

### 2. Create a Python environment

Recommended Python versions:

- Python 3.11
- Python 3.12

Example using `venv`:

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Example using `conda`:

```bash
conda create -n salifort-streamlit python=3.12 -y
conda activate salifort-streamlit
```

### 3. Install runtime dependencies

```bash
pip install -r requirements.txt
```

Current runtime dependencies are defined in `requirements.txt` and support the hosted app itself. They are not meant to reproduce every offline modeling dependency used by the builders. The OpenAI Python package is included for optional hosted reviewer briefings that use compact aggregate summaries only. `scikit-learn`, `xgboost`, and `joblib` are included so Streamlit Cloud can load the committed packaged MLOps Lab demo model artifact under `artifacts/mlops_lab_online/`.

The optional MLOps Mini-Lab dependencies live in `requirements-mlops.txt`. They are for local/dev data prep, training, FastAPI serving, MLflow tracking, and tests. They are not required for the standard Streamlit app runtime.

## Running the App Locally

Start the Streamlit app from the repo root:

```bash
python -m streamlit run app/app.py
```

Equivalent shortcut:

```bash
streamlit run app/app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

To stop the app, return to the terminal and press `Ctrl + C`.

## Optional Local Docker Demo

The MLOps Mini-Lab includes an optional Docker Compose demo for local development. It can run the FastAPI lab-serving service, the existing Streamlit app, and an optional MLflow UI without changing the public Streamlit artifact truth.

See the [Docker local runbook](../mlops-docker-local-runbook.md) for commands and service boundaries.

## Optional Local Airflow DAG Scaffold

The MLOps Mini-Lab also includes an optional Airflow DAG scaffold for local development. It orchestrates the lab CLI scripts outside Streamlit and does not update public app artifacts.

See the [Airflow local runbook](../mlops-airflow-local-runbook.md) for setup and validation notes.

## MLOps Lab Page Boundary

The Streamlit MLOps Lab page is a read-only reviewer surface. It can show local file status, report summaries, command examples, and optional FastAPI health/model-info checks. It does not run training, start Docker, trigger Airflow, run MLflow, or change public model artifacts.

## Optional OpenAI API Setup for Advanced Reviewer Features

Most of the app can run without an API key.

Some advanced PACE Navigator reviewer features use OpenAI embeddings for retrieval-backed evidence review. If you want those features to work locally, provide one of these environment variables:

- `RAG_STREAMLIT_OPENAI_API_KEY`
- `OPENAI_API_KEY`

### Windows PowerShell

```powershell
$env:RAG_STREAMLIT_OPENAI_API_KEY="your_api_key_here"
```

### macOS / Linux

```bash
export RAG_STREAMLIT_OPENAI_API_KEY="your_api_key_here"
```

Important:

- do not hardcode API keys in source files
- do not commit keys to the repo
- do not add keys to public documentation examples beyond environment-variable placeholders

## Important Project Directories

- `app/`: Streamlit app and page code
- `data/`: checked-in HR dataset
- `artifacts/v2/`: generated model and explanation artifacts
- `outputs/figures/`: stable figure assets
- `navigator/`: advanced reviewer registries, retrieval assets, workflow contracts, and readiness metadata
- `scripts/`: offline build and validation scripts
- `docs/`: documentation and walkthroughs

## Offline Builders and Validators

This repo includes offline scripts for:

- building `artifacts/v2/`
- building the retrieval pack
- building the retrieval index
- validating contracts, retrieval runtime, answer assembly, and readiness

Those scripts live under `scripts/`. They are useful for local maintenance and technical review, but Streamlit does not run them automatically in the app.

## Streamlit Community Cloud Deployment

### 1. Push the repo to GitHub

Make sure the deployment branch is pushed and contains:

- `requirements.txt`
- `app/app.py`
- all required checked-in data/artifacts/docs you want available in the deployment

### 2. Create a new app in Streamlit Community Cloud

In Streamlit Community Cloud:

1. choose the GitHub repository
2. choose the target branch
3. set the entrypoint file to:

```text
app/app.py
```

Streamlit Community Cloud deployment still uses `requirements.txt` and `app/app.py`. Docker Compose, FastAPI, MLflow, Airflow, and `requirements-mlops.txt` are local/dev extension tools and are not required for the public app to open.

### 3. Configure secrets only if needed

If the deployment should support advanced retrieval-backed reviewer features, add the API key as a Streamlit secret or environment variable through the hosting platform.

Example secret:

```toml
RAG_STREAMLIT_OPENAI_API_KEY = "your_api_key_here"
```

If the hosted MLOps Lab should generate optional aggregate briefings for the Online CSV Insight sandbox, add:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_SUMMARY_MODEL = "gpt-5.4-mini"
```

The hosted Online CSV Insight sandbox runs directly in Streamlit Cloud. It supports transparent heuristic scoring and packaged demo model inference from the committed artifact in `artifacts/mlops_lab_online/`. It does not require `SALIFORT_API_URL`, `SALIFORT_API_TOKEN`, FastAPI, Docker, MLflow, Airflow, Render, or any external scoring API. It sends only compact aggregate JSON to OpenAI for optional briefings; raw uploaded CSV rows and identifier-like fields are not sent.

### 4. Deploy and smoke-test

After deployment, verify:

- Overview loads correctly
- Workforce Explorer and Model & Threshold Lab render correctly
- Explainability and Manager Action View show their expected artifacts
- Methods & Limitations renders without missing content
- PACE Navigator loads
- MLOps Lab loads, including Online CSV Insight, MLOps Evidence, and Responsible Use tabs
- advanced retrieval-backed features either work correctly with a key or show a clear blocked/setup-needed state without one
- optional Online CSV Insight AI briefing works only when `OPENAI_API_KEY` is configured and uses compact aggregate JSON only

## Deployment Boundaries

This repo should not be described as:

- a production HR platform
- a live Airflow deployment
- a system that runs workflows from Streamlit
- a fully autonomous agent application

The accurate posture is:

- a public portfolio/demo Streamlit app
- with optional advanced reviewer features
- and explicit runtime boundaries around model training, retrieval, workflow review, and secrets handling

## Hosted MLOps Lab CSV Insight

The MLOps Lab includes a hosted Streamlit-only CSV Insight sandbox. Visitors can upload a small Salifort-style CSV, validate the schema, run a transparent review-priority heuristic in pandas, run packaged demo model inference when the committed online artifact is present, download a review summary CSV, and optionally generate an OpenAI-assisted briefing from compact aggregate statistics.

This hosted path does not deploy or call FastAPI. Local/dev FastAPI, Docker Compose, MLflow, and Airflow remain valid technical review components, but they are not required for Streamlit Community Cloud and are not run inside visitor sessions.

## Optional Local MLOps Commands

Install local/dev tooling only when you want to reproduce the MLOps Mini-Lab:

```bash
pip install -r requirements-mlops.txt
python scripts/mlops_run_pipeline.py
python scripts/export_mlops_evidence_pack.py
python scripts/export_streamlit_model_artifact.py
python -m uvicorn api.main:app --reload
mlflow ui
docker compose config
docker compose up api
docker compose up streamlit
docker compose --profile mlflow up mlflow
python scripts/validate_mlops_airflow_dag.py
```

These commands are for a development machine. Streamlit Community Cloud does not run training, FastAPI, Docker, MLflow, or Airflow for visitor sessions.

## Troubleshooting

### App does not start

Check:

- the environment is activated
- dependencies are installed
- you are running from the repo root
- the command is `streamlit run app/app.py`

### Import or dependency errors

Reinstall the runtime dependencies:

```bash
pip install -r requirements.txt
```

### Advanced retrieval-backed features are unavailable

Check:

- whether an API key is present in the local environment or deployment secrets
- whether the retrieval pack and retrieval index files exist in the repo/deployment

### Local app works but deployment is missing features

Check:

- the correct branch was deployed
- `app/app.py` is the configured entry point
- required artifacts were included in the deployed branch
- secrets were set in the hosting environment when needed

## Related Documents

- [Root README](../../README.md)
- [Documentation Guide](../README.md)
- [Technical Design and Architecture](../technical/technical-design-and-architecture.md)
- [User Manual](../user-guide/user-manual.md)
- [Streamlit App Walkthrough](../user-guide/streamlit-app-walkthrough.md)
- [MLOps Mini-Lab Demo Guide](../mlops-demo-guide.md)
- [Formal Documentation Package](../formal/salifort-formal-document-package.md)
