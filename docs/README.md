# Documentation Guide

This folder is the documentation hub for the **Salifort Motors Retention Risk Explorer**.

Use the root [README](../README.md) for the short project overview. Use this guide when you want the app walkthrough, product notes, technical design, local/dev MLOps runbooks, Navigator notes, or evidence references.

## Recommended Reading Paths

### First project review

1. [Root README](../README.md)
2. [Streamlit App Walkthrough](user-guide/streamlit-app-walkthrough.md)
3. [User Manual](user-guide/user-manual.md)
4. [Executive Summary](executive/executive-summary.md)

### Business and user guidance

1. [Executive Summary](executive/executive-summary.md)
2. [HR Quick Start](user-guide/hr-quick-start.md)
3. [User Manual](user-guide/user-manual.md)
4. [Product Requirements Document](product/product-requirements-document.md)

### Technical review

1. [Technical Design and Architecture](technical/technical-design-and-architecture.md)
2. [Environment Setup and Deployment Guide](deployment/environment-setup-and-deployment-guide.md)
3. [Navigator Notes](navigator/README.md)
4. [MLOps Mini-Lab Demo Guide](mlops-demo-guide.md)

### Local/dev MLOps review

1. [MLOps Mini-Lab Demo Guide](mlops-demo-guide.md)
2. [MLOps Docker Local Runbook](mlops-docker-local-runbook.md)
3. [MLOps Airflow Local Runbook](mlops-airflow-local-runbook.md)
4. [MLOps CI Runbook](mlops-ci-runbook.md)
5. [MLOps Evidence Pack](demo-assets/mlops-evidence/README.md)

### Responsible-use and repository guidance

1. [Responsible-Use Eval Cases](../evals/README.md)
2. [Repository Agent Guidance](../AGENTS.md)
3. [Repo-Scoped Skills](../.agents/skills/)

## Core Documents

### App and user guides

- [Streamlit App Walkthrough](user-guide/streamlit-app-walkthrough.md): page-by-page flow through the nine-page app.
- [User Manual](user-guide/user-manual.md): user-facing guidance for interpreting app pages responsibly.
- [HR Quick Start](user-guide/hr-quick-start.md): concise business guide for HR-style and manager-style review.
- [Executive Summary](executive/executive-summary.md): non-technical management summary and responsible-use framing.

### Product and technical docs

- [Product Requirements Document](product/product-requirements-document.md): product goals, audiences, scope, non-goals, and success criteria.
- [Technical Design and Architecture](technical/technical-design-and-architecture.md): runtime layers, data flow, artifacts, retrieval design, workflow contracts, and deployment boundaries.
- [Environment Setup and Deployment Guide](deployment/environment-setup-and-deployment-guide.md): local setup, optional OpenAI configuration, Streamlit Community Cloud deployment, and hosted/local-dev boundaries.
- [Formal Documentation Package](formal/salifort-formal-document-package.md): formal document index and maintenance notes.

### MLOps local/dev docs

- [MLOps Mini-Lab Demo Guide](mlops-demo-guide.md): hosted CSV Insight walkthrough plus local/dev pipeline, API, Docker, MLflow, Airflow, CI, and evidence review.
- [MLOps Docker Local Runbook](mlops-docker-local-runbook.md): optional local/dev Docker Compose demo for API, Streamlit, and MLflow services.
- [MLOps Airflow Local Runbook](mlops-airflow-local-runbook.md): optional local/dev Airflow DAG scaffold for lab CLI scripts.
- [MLOps CI Runbook](mlops-ci-runbook.md): GitHub Actions checks for app runtime, MLOps tests, Airflow validation, and Docker Compose config.
- [MLOps Evidence Pack](demo-assets/mlops-evidence/README.md): committed, sanitized proof snapshots for local/dev MLOps review.

### Navigator docs

- [Navigator documentation](navigator/README.md): overview of the PACE Navigator governed review layer.
- [Demo readiness walkthrough](navigator/demo-readiness-walkthrough.md): current Navigator demo posture and boundaries.
- `docs/navigator/wp*.md`: implementation-history notes for the Navigator build-out.

### Evidence and repository guides

- [Artifacts guide](../artifacts/v2/README.md)
- [Data guide](../data/README.md)
- [Figures guide](../outputs/figures/README.md)
- [Navigator metadata guide](../navigator/README.md)
- [Scripts guide](../scripts/README.md)
- [Responsible-Use Eval Cases](../evals/README.md)

## Scope Notes

- The root `README.md` is the concise public entry point.
- `docs/README.md` is the canonical documentation index.
- `docs/user-guide/streamlit-app-walkthrough.md` is the canonical page-by-page app walkthrough.
- `docs/streamlit-app-walkthrough.md` is a compatibility redirect to the canonical walkthrough.
- `docs/navigator/wp*.md` files are implementation-history notes and should not replace the PRD, technical design, or user manual.
- MLOps runbooks are local/dev guides. Hosted Streamlit does not require `SALIFORT_API_URL`, `SALIFORT_API_TOKEN`, FastAPI, Docker, MLflow, Airflow, Render, or local pipeline outputs.

## Documentation Principles

- Describe the repo as a portfolio Streamlit decision-support app, not a production HR platform.
- Keep the public reference model threshold `0.29` separate from the MLOps Lab packaged demo threshold `0.60`.
- Describe retrieval, OpenAI usage, MLOps Lab, FastAPI, Docker, MLflow, Airflow, and Agent Shell behavior proportionally.
- Treat the PACE Navigator as a governed review surface for fixed review questions, not arbitrary chat prompts.
- Treat the Agent Shell as preview-only plan mapping that does not execute workflows or background jobs.
- Keep OpenAI briefing language aggregate-only: raw uploaded CSV rows and identifier-like fields are not sent to OpenAI.
