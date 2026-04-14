# Salifort Retention Risk Explorer (Streamlit)

Deployment-focused Streamlit V1 for the Salifort Motors portfolio project.

## Live app

https://salifort-retention-risk-explorer.streamlit.app/

## What this repo is

This repository contains the deployment-focused Streamlit V1 of the Salifort Motors portfolio project. It presents the operational HR analytics decision app layer using the repo-local CSV and existing checked-in project visuals.

The interactive Workforce Explorer uses a lightweight V1 screening proxy for responsive filtering and summaries. The weighted XGBoost model story, threshold logic, and SHAP visuals shown in the app come from the original portfolio workflow and checked-in artifacts.

## What this app shows

This app presents the public operational version of the project as a lightweight decision-support layer.

Main themes:
- cost-aware retention screening
- workforce exploration
- workload and tenure risk patterns
- threshold trade-offs
- SHAP-based explainability
- department exposure and manager action view

## Main dataset
- `data/hr_capstone_dataset.csv`

## Main figures
- `outputs/figures/`

## Local run
```bash
streamlit run app/app.py
```
