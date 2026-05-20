# Salifort Motors Retention Risk Explorer

Portfolio Streamlit app for retention-risk analysis, model interpretation, responsible-use review, and local/dev MLOps evidence.

Live app: https://salifort-retention-risk-explorer.streamlit.app/

This project is a portfolio Streamlit decision-support app. It is not a production HR platform, not an employment decision system, and not a live enterprise deployment.

## Business Problem

Salifort Motors wants to understand which workforce patterns may be associated with employee attrition risk, where manager review should start, and how to keep model outputs from becoming automated HR decisions.

The app uses a Salifort-style HR capstone dataset to support a careful review workflow:

- explore workforce patterns by department, salary, tenure, workload, and risk flag
- compare model and threshold trade-offs
- interpret model behavior through SHAP outputs
- review department exposure before jumping to individual conclusions
- keep limitations and responsible-use boundaries visible

## Methods and Architecture

The hosted app runs from `app/app.py` and reads committed files from the repository. It does not retrain models, start services, run shell commands, trigger Airflow, or mutate artifacts during a visitor session.

The project is organized into four main layers:

- `data/hr_capstone_dataset.csv` provides the checked-in demo dataset.
- `artifacts/v2/` provides the public app model metadata, row-level scores, threshold tables, department exposure, and SHAP summaries.
- `outputs/figures/` provides stable figures used by the app and documentation.
- `navigator/` and `app/services/` support the governed PACE Navigator review layer.

The local/dev MLOps mini-lab is separate from the hosted app. It includes pipeline scripts, FastAPI, MLflow, Docker Compose, Airflow scaffolds, CI checks, and evidence export for technical review. These pieces are optional local/dev surfaces, not hosted production infrastructure.

## Key Features and Evidence

The Streamlit sidebar contains nine pages:

- **Overview:** project framing, dataset summary, public model truth, and responsible-use boundaries
- **PACE Navigator:** governed project review surface with fixed questions, citations, source preview, workflow readiness, and preview-only plan mapping
- **Workforce Explorer:** interactive workforce filters and department exposure review
- **EDA & Patterns:** stable visual summaries of workforce patterns
- **Model & Threshold Lab:** model comparison, threshold trade-offs, precision, recall, and review workload
- **Explainability:** SHAP-based model behavior review without causal claims
- **Manager Action View:** department-level review priorities and manager-facing interpretation
- **Methods & Limitations:** architecture, artifact, privacy, fallback, retrieval, and production-boundary notes
- **MLOps Lab:** hosted Online CSV Insight, packaged demo model scoring, MLOps Evidence Pack, and local/dev MLOps review surfaces

The hosted MLOps Lab includes Online CSV Insight for small Salifort-style uploads. It can run transparent heuristic scoring, optional packaged demo model scoring, and optional OpenAI-assisted briefings from compact aggregate JSON only. Raw uploaded CSV rows and identifier-like fields are not sent to OpenAI.

## Model and Artifact Truth

The public reference model story is governed by `artifacts/v2/metadata.json`:

- public reference model: weighted XGBoost
- selected public threshold: `0.29`
- runtime posture: load committed artifacts when available; do not train inside Streamlit

Some views can fall back to a simpler screening score if row-level artifacts are missing. That fallback keeps the app explorable, but it is not the final model probability.

The MLOps Lab packaged demo model is governed separately by `artifacts/mlops_lab_online/model_metadata.json`:

- model scope: MLOps Lab online demo
- lab threshold: `0.60`
- purpose: hosted Streamlit demo scoring for Online CSV Insight

The lab model and threshold do not replace the public weighted XGBoost threshold `0.29` story.

## Responsible-Use Boundaries

This project should be read as a review and learning tool, not as an HR operating system.

Important boundaries:

- model outputs are review cues, not employment decisions
- SHAP explains model behavior and associations, not causal proof
- hosted Streamlit does not require FastAPI, Docker, MLflow, Airflow, Render, or `SALIFORT_API_URL`
- local/dev MLOps components are optional technical review surfaces
- the PACE Navigator is a governed review surface that does not answer arbitrary chat prompts
- the Agent Shell is preview-only and does not execute workflows, trigger background jobs, or act autonomously
- OpenAI features are optional and use either embeddings for governed retrieval or aggregate-only briefing payloads

## Local Run

Install the hosted app dependencies:

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Optional retrieval-backed Navigator features may require a local OpenAI API key supplied through `RAG_STREAMLIT_OPENAI_API_KEY` or `OPENAI_API_KEY`. Do not commit API keys or secrets.

Optional local/dev MLOps tooling uses a separate dependency file:

```bash
pip install -r requirements-mlops.txt
python scripts/mlops_run_pipeline.py
python scripts/export_mlops_evidence_pack.py
python scripts/export_streamlit_model_artifact.py
python scripts/validate_mlops_airflow_dag.py
```

Optional local services such as FastAPI, Docker Compose, MLflow, and Airflow are documented in the MLOps runbooks. They are not required for the hosted Streamlit app.

## Documentation

Start with:

- [Documentation Guide](docs/README.md)
- [Streamlit App Walkthrough](docs/user-guide/streamlit-app-walkthrough.md)
- [User Manual](docs/user-guide/user-manual.md)
- [Technical Design and Architecture](docs/technical/technical-design-and-architecture.md)
- [MLOps Mini-Lab Demo Guide](docs/mlops-demo-guide.md)
- [Navigator Notes](docs/navigator/README.md)

## Related Files

- [app/app.py](app/app.py)
- [app/pages/](app/pages/)
- [data/hr_capstone_dataset.csv](data/hr_capstone_dataset.csv)
- [artifacts/v2/](artifacts/v2/)
- [artifacts/mlops_lab_online/](artifacts/mlops_lab_online/)
- [outputs/figures/](outputs/figures/)
- [docs/README.md](docs/README.md)
- [docs/user-guide/streamlit-app-walkthrough.md](docs/user-guide/streamlit-app-walkthrough.md)
- [docs/mlops-demo-guide.md](docs/mlops-demo-guide.md)
- [docs/navigator/README.md](docs/navigator/README.md)
- [requirements.txt](requirements.txt)
- [requirements-mlops.txt](requirements-mlops.txt)
