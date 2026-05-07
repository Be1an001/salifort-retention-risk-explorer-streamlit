# Repository Guidance for Coding Agents

## Project Purpose

This repository contains the Salifort Motors Retention Risk Explorer: a portfolio Streamlit app for HR analytics and retention-risk decision support. The app is not a production HR system, not an employment decision system, and not an autonomous agent runtime.

The deployed product remains a nine-page Streamlit app with artifact-backed public model results, Online CSV Insight, packaged MLOps Lab demo model inference, optional aggregate-only OpenAI briefings, PACE Navigator review surfaces, and local/dev MLOps evidence.

## Repository Layout

- `app/app.py`: Streamlit entry point.
- `app/pages/`: nine Streamlit pages.
- `app/pages/mlops_lab.py`: Online CSV Insight, packaged demo model scoring, optional aggregate AI briefing, MLOps Evidence Pack display, and local/dev MLOps review surfaces.
- `app/pages/methods_limitations.py`: methods, limitations, runtime boundaries, and responsible-use copy.
- `app/pages/pace_navigator.py`: governed reviewer surfaces, fixed-question retrieval review, source preview, audit exports, workflow readiness, and preview-only plan mapping.
- `app/utils/`: data and artifact loading helpers.
- `app/services/` and `app/viewmodels/`: PACE Navigator service and presentation helpers.
- `data/`: checked-in Salifort-style HR dataset.
- `artifacts/v2/`: public app artifacts and public model truth.
- `artifacts/mlops_lab_online/`: committed packaged demo model artifact for hosted MLOps Lab inference.
- `outputs/figures/`: stable PNG figures used by the app and docs.
- `docs/`: product, technical, deployment, user, executive, formal, Navigator, and MLOps runbook documentation.
- `navigator/`: PACE Navigator registries, retrieval assets, readiness contracts, and governed source metadata.
- `src/salifort_mlops/`, `api/`, `mlops/`, `docker/`, and `orchestration/airflow/dags/`: local/dev MLOps Mini-Lab modules, service, generated-output placeholders, Docker files, and DAG scaffolds.
- `scripts/`: artifact builders, MLOps scripts, evidence exporters, and validators.
- `tests/`: data, app, API, MLOps, Airflow, Online CSV Insight, and evidence-pack contract tests.
- `.agents/skills/`: repo-scoped Codex skills for repeatable review workflows.
- `evals/`: lightweight JSONL cases for AI briefing and responsible-use behavior review.

## Hosted vs Local/Dev Split

Hosted Streamlit:

- runs from `app/app.py`
- reads checked-in data, artifacts, docs, Navigator metadata, static figures, and the packaged MLOps Lab demo model
- supports Online CSV Insight directly in Streamlit
- does not require `SALIFORT_API_URL`, `SALIFORT_API_TOKEN`, Render, FastAPI, Docker, MLflow, or Airflow
- sends only compact aggregate JSON to OpenAI for optional briefings

Local/dev MLOps:

- may run `scripts/mlops_run_pipeline.py`, FastAPI, MLflow, Docker Compose, and Airflow DAG validation
- writes generated lab outputs under `mlops/data`, `mlops/models`, `mlops/reports`, and `mlruns/`
- does not update public app artifacts unless an explicit artifact export task is requested

## Model and Artifact Truth

- Public app truth: weighted XGBoost at threshold `0.29`, governed by `artifacts/v2/metadata.json`.
- MLOps Lab packaged demo truth: lab demo model at threshold `0.60`, governed by `artifacts/mlops_lab_online/model_metadata.json`.
- Keep these truths separate in docs, tests, and page copy.
- Do not modify `artifacts/v2/` unless the task explicitly asks for a governed artifact update.
- Do not modify `artifacts/mlops_lab_online/champion_model.joblib` or retrain/export model artifacts unless explicitly requested.

## Responsible-Use Boundaries

Always preserve these boundaries:

- This is a portfolio-grade decision-support demo.
- This is not a production HR platform.
- This is not an employment decision system.
- Outputs support human review only.
- SHAP and model outputs explain associations, not causal proof.
- OpenAI optional briefings receive compact aggregate summaries only.
- Raw uploaded CSV rows, row-level employee data, and PII-like fields must not be sent to OpenAI.
- Streamlit must not run training, Docker, MLflow, Airflow, git, CI, shell commands, or background jobs from a visitor session.
- The PACE Navigator agent shell is preview-only and must not execute workflows.
- Do not add AgentKit, Agents SDK, MCP, Responses API runtime, or production agent architecture unless the user explicitly asks for that separate feature.

## Key Commands

Run the app locally:

```bash
python -m streamlit run app/app.py
```

Install app dependencies:

```bash
python -m pip install -r requirements.txt
```

Install local/dev MLOps dependencies:

```bash
python -m pip install -r requirements-mlops.txt
```

Safe validation commands:

```bash
python -m py_compile app/app.py app/pages/mlops_lab.py app/pages/methods_limitations.py
python -m pytest tests/test_online_csv_insight_contract.py tests/test_mlops_evidence_pack_contract.py
python scripts/validate_mlops_airflow_dag.py
docker compose config
docker compose --profile mlflow config
```

Full local/dev MLOps commands, only when specifically needed:

```bash
python scripts/mlops_run_pipeline.py
python scripts/export_mlops_evidence_pack.py
python scripts/export_streamlit_model_artifact.py
python -m uvicorn api.main:app --reload
mlflow ui
docker compose up api
docker compose up streamlit
docker compose --profile mlflow up mlflow
```

Do not run Streamlit, FastAPI, MLflow, Docker services, Airflow, or training pipelines unless the user asks or the task requires it.

## Files That Must Not Be Committed

Do not stage or commit:

- secrets, API keys, `.env`, or `.streamlit/secrets.toml`
- uploaded CSVs or user-provided data
- `mlruns/`
- `.pytest_cache/`, `__pycache__/`, `.venv/`, or local caches
- generated lab outputs under `mlops/data`, `mlops/models`, or `mlops/reports`, except tracked `.gitkeep` placeholders
- generated model binaries unless explicitly requested

Before committing, inspect `git status --short` and staged files.

## Done Definition

A change is done when:

- the requested scope is complete without unrelated app logic changes
- hosted/local-dev boundaries remain accurate
- public threshold `0.29` and lab threshold `0.60` remain separated
- responsible-use and OpenAI privacy boundaries remain intact
- app logic and artifacts are untouched unless explicitly requested
- safe validation commands have been run or any skipped command is clearly explained
- no secrets, caches, generated outputs, or uploaded data are staged
- final notes summarize files changed, validation results, and remaining risks
