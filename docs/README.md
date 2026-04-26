# Documentation Guide

This folder is the documentation hub for the **Salifort Motors Retention Risk Explorer**.

Use it when you want more detail than the root `README.md` provides.

## Recommended Reading Paths

### For GitHub visitors and interviewers

1. [Root README](../README.md)
2. [Streamlit App Walkthrough](user-guide/streamlit-app-walkthrough.md)
3. [User Manual](user-guide/user-manual.md)

### For technical reviewers

1. [Technical Design and Architecture](technical/technical-design-and-architecture.md)
2. [Navigator Notes](navigator/README.md)
3. [Demo Readiness Walkthrough](navigator/demo-readiness-walkthrough.md)

### For developers who want to run or deploy the app

1. [Environment Setup and Deployment Guide](deployment/environment-setup-and-deployment-guide.md)
2. [Technical Design and Architecture](technical/technical-design-and-architecture.md)
3. [Scripts Folder Guide](../scripts/README.md)
4. [MLOps Docker Local Runbook](mlops-docker-local-runbook.md)
5. [MLOps Airflow Local Runbook](mlops-airflow-local-runbook.md)
6. [MLOps CI Runbook](mlops-ci-runbook.md)
7. [MLOps Mini-Lab Demo Guide](mlops-demo-guide.md)

## Core Documents

### Product and project framing

- [Product Requirements Document](product/product-requirements-document.md): product goals, audiences, scope, non-goals, and success criteria.

### Technical design

- [Technical Design and Architecture](technical/technical-design-and-architecture.md): runtime layers, data flow, artifacts, retrieval design, workflow contracts, and deployment boundaries.

### Setup and deployment

- [Environment Setup and Deployment Guide](deployment/environment-setup-and-deployment-guide.md): local setup, optional API configuration, runtime expectations, and Streamlit Community Cloud deployment.
- [MLOps Docker Local Runbook](mlops-docker-local-runbook.md): optional local/dev Docker Compose demo for API, Streamlit, and MLflow services.
- [MLOps Airflow Local Runbook](mlops-airflow-local-runbook.md): optional local/dev Airflow DAG scaffold for orchestrating lab CLI scripts.
- [MLOps CI Runbook](mlops-ci-runbook.md): GitHub Actions checks for app runtime, MLOps tests, Airflow static validation, and Docker Compose config.
- [MLOps Mini-Lab Demo Guide](mlops-demo-guide.md): hosted CSV Insight walkthrough plus local/dev pipeline, API, Docker, MLflow, Airflow, and CI evidence path.

### User-facing guidance

- [User Manual](user-guide/user-manual.md): how to use the app responsibly and what each page is for.
- [Streamlit App Walkthrough](user-guide/streamlit-app-walkthrough.md): page-by-page review flow and what to look for.
- MLOps Lab is covered in the walkthrough as an optional technical-review page for the local/dev extension.
- The [MLOps Mini-Lab Demo Guide](mlops-demo-guide.md) gives a concise reviewer script for the hosted Online CSV Insight sandbox and local/dev evidence.

## Advanced Navigator Documentation

- [Navigator documentation](navigator/README.md): overview of the advanced PACE Navigator reviewer layer.
- [Demo readiness walkthrough](navigator/demo-readiness-walkthrough.md): current demo posture and review boundaries.
- `docs/navigator/wp*.md`: technical implementation-history notes for the advanced Navigator layers.

## Existing Folder Guides

- [Artifacts guide](../artifacts/v2/README.md)
- [Data guide](../data/README.md)
- [Figures guide](../outputs/figures/README.md)
- [Navigator metadata guide](../navigator/README.md)
- [Scripts guide](../scripts/README.md)

## Documentation Principles in This Repo

- The root `README.md` stays the main entry point.
- Formal docs live under `docs/`.
- Advanced Navigator notes remain available, but they are not the first stop for most visitors.
- The repo is described as a portfolio and educational decision-support app, not a production HR platform.
- Retrieval, OpenAI usage, Airflow, and the agent shell are documented accurately and proportionally.
