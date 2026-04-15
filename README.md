# Salifort Retention Risk Explorer (Streamlit)

Streamlit app for the Salifort Motors retention risk project.

## Live app

https://salifort-retention-risk-explorer.streamlit.app/

## Project question

How can Salifort Motors identify attrition exposure early enough to support targeted, responsible retention action without turning the workflow into an automated HR decision engine?

## What this repo contains

This repository contains the public app layer for the Salifort Motors project. It uses the checked-in HR dataset, generated model tables, and existing project visuals to present an operational decision-support view of attrition risk.

The Workforce Explorer supports interactive review of employee and department patterns. Model comparison, threshold analysis, department exposure, and explainability are powered by generated artifacts built outside Streamlit runtime.

## What this app shows

This app presents the project as a lightweight decision-support tool for reviewing attrition risk.

Main themes:
- threshold-based retention screening
- workforce exploration
- workload and tenure risk patterns
- threshold trade-offs
- SHAP-based explainability
- department exposure and manager action view

## Main files

- `data/hr_capstone_dataset.csv`
- `outputs/figures/`
- `artifacts/v2/`

## Notes and limitations

This is an educational project, not a live HR system. It is meant to support review and discussion, not automate employment decisions. Model training and artifact generation happen outside Streamlit runtime, and the app reads generated outputs when they are available.

## Local run
```bash
streamlit run app/app.py
```
