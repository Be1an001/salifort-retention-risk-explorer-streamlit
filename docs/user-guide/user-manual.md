# User Manual

## Document Purpose

This manual is for people who want to use the **Salifort Motors Retention Risk Explorer** as a review and discussion tool.

It is especially useful for:

- HR-style readers
- manager-style readers
- portfolio and technical reviewers
- anyone who wants a clear guide before opening the more technical Navigator features

This app supports review and discussion. It should not be treated as an automated employment decision system.

One-sentence system intro: the app is a portfolio Streamlit decision-support demo that helps reviewers explore employee retention-risk patterns, model trade-offs, explanations, and MLOps evidence without automating HR decisions.

## What the App Helps You Do

The app helps you:

- review the project question and model story
- explore workforce slices by department, salary, tenure, and risk flags
- understand threshold trade-offs
- inspect feature-level explanation through SHAP outputs
- review department exposure and manager-oriented priorities
- understand technical and responsible-use boundaries
- inspect the MLOps Lab for hosted CSV insight, packaged demo model inference, and local/dev MLOps evidence

It also includes optional advanced review tooling in the PACE Navigator for visitors who want to inspect evidence, citations, workflow structure, and readiness details.

## App Layout

### Sidebar and navigation

Use the left sidebar to move across the nine Streamlit pages. The pages are grouped by review intent:

- **Start Here:** Overview and PACE Navigator.
- **Interactive Analysis:** Workforce Explorer and EDA & Patterns.
- **Decision Support:** Model & Threshold Lab, Explainability, Manager Action View, and Methods & Limitations.
- **Governance & Ops:** MLOps Lab.

The sidebar tells you where you are in the product. It does not change data, retrain models, or run MLOps commands.

### Main panel

The main panel is where each page shows filters, charts, tables, interpretation notes, and responsible-use warnings. Read the captions and warning boxes closely because they explain whether a view is using generated model artifacts, fallback screening logic, hosted CSV sandbox scoring, or local/dev MLOps evidence.

## Suggested Reading Order

For most first-time visitors, this order works best:

1. Overview
2. Workforce Explorer
3. EDA & Patterns
4. Model & Threshold Lab
5. Explainability
6. Manager Action View
7. Methods & Limitations
8. PACE Navigator if you want the advanced reviewer layer
9. MLOps Lab if you want hosted CSV insight and MLOps evidence

## Page-by-Page Guide

### Overview

Use this page to understand:

- what the project is
- which data and artifacts it uses
- what the public model truth is
- what the threshold truth is
- what the app does and does not do

Best use:

- start here before opening any other page

### Workforce Explorer

Use this page to review workforce slices and department patterns.

You can usually filter by:

- department
- salary band
- tenure band
- selected-threshold or fallback high-risk flag

Best use:

- compare how risk signals change across different groups
- inspect department-level exposure and employee-level review rows

Interpret carefully:

- generated row-level artifacts reflect the preferred runtime truth when available
- fallback screening keeps the app usable but is not the final model probability

### EDA & Patterns

Use this page to understand the broader workforce story behind the model.

Best use:

- review the stable visual patterns before discussing model outputs in isolation

Look for:

- workload and satisfaction clusters
- department and salary differences
- tenure and project-load combinations that may signal pressure

### Model & Threshold Lab

Use this page to understand model and threshold trade-offs.

Best use:

- explain why threshold `0.29` matters
- compare recall, precision, flagged review volume, and confusion-matrix behavior

Interpret carefully:

- threshold choice is a review-design decision, not just a technical metric
- accuracy alone is not enough in an attrition-risk setting

Business-language metrics:

- **Recall:** among employees who actually left in the test data, how many the model caught for review.
- **Precision:** among employees flagged for review, how many were actual leavers in the test data.
- **Threshold:** the probability cutoff used to decide how many cases should enter review.
- Lowering the threshold usually catches more true leavers but also sends more people into review who may not leave.

### Explainability

Use this page to understand which features most influence the model signal.

Best use:

- explain why the model flags certain kinds of cases
- support review discussions with feature-level evidence

Interpret carefully:

- SHAP explains model behavior
- SHAP does not prove why an employee left
- feature importance is a guide to what the model relied on, not a causal HR finding

### Manager Action View

Use this page to translate analysis into review priorities.

Best use:

- compare total exposure and normalized exposure by department
- discuss which teams deserve earlier review attention

Interpret carefully:

- this page supports prioritization and discussion
- it should not be used to punish or rank employees automatically

### Methods & Limitations

Use this page to understand how the app works and where its boundaries are.

It explains:

- checked-in data
- duplicate removal and cleaning
- generated artifacts
- static figures
- fallback logic
- retrieval/RAG
- PACE
- workflow contracts, Airflow scaffold, and preview-only agent shell

### PACE Navigator

Use this page only after you understand the main app story.

What it offers:

- a guided project map
- source-of-truth and drift drilldowns
- fixed-question answer viewing with retrieval-backed evidence
- citations, source detail, and governed source preview
- workflow and readiness review
- preview-only plan routing

What it does not do:

- it does not answer arbitrary chat prompts
- it does not run Airflow jobs
- it does not execute workflows or act autonomously

### MLOps Lab

Use this page if you want to review the local/dev technical extension.

What it offers:

- hosted Online CSV Insight for small Salifort-style uploads
- transparent heuristic review scoring
- packaged demo model inference from `artifacts/mlops_lab_online/`
- optional OpenAI briefing from compact aggregate JSON only
- committed MLOps Evidence Pack snapshots
- local/dev CLI, MLflow, FastAPI, Docker, Airflow, and CI guidance

What it does not do:

- it does not replace the public weighted XGBoost threshold `0.29` model truth
- it does not require FastAPI, Docker, MLflow, Airflow, Render, or external scoring APIs for hosted use
- it does not run training, workflows, shell commands, Docker, Airflow, MLflow, git, or CI from Streamlit
- it does not send raw uploaded CSV rows or identifier-like fields to OpenAI

## How to Interpret Key Ideas

### Public model truth

The public reference model is:

- **weighted XGBoost**
- **selected threshold: `0.29`**

### Fallback logic

Fallback logic exists so the app remains explorable when some generated artifacts are missing. It is useful for resilience, but it is not the same thing as the final model probability.

### Threshold

The threshold controls how broadly the app flags likely risk cases for review. Lower thresholds can capture more true positives but can also increase false positives and review workload.

In the public app, the selected threshold is `0.29` for the public weighted XGBoost story. In the MLOps Lab packaged demo model, the lab threshold is separate and currently `0.60`.

### Risk probability

A risk probability is a model score between 0 and 1. Higher values mean the model sees a stronger similarity to employees who left in the training history. It is a review cue, not a decision.

### High / Medium / Low bands

Where the app shows High, Medium, or Low bands, use them as prioritization labels:

- **High:** review earlier and look for workload, promotion, salary, manager, or team-context issues.
- **Medium:** watch and compare with department context before escalating.
- **Low:** no immediate model-driven priority, but still consider normal HR context.

Online CSV Insight heuristic bands come from transparent rules. Packaged demo model bands come from the lab model threshold. Neither banding system replaces the public app threshold `0.29`.

### Explainability

SHAP helps explain what the model is responding to. It should be used to guide discussion, not to make causal claims.

### Retrieval-backed review

Some PACE Navigator features use a prepared retrieval corpus and optional OpenAI embeddings to help a reviewer inspect evidence behind fixed review questions. This is review support, not arbitrary AI chat.

## Recommended Responsible Use

- Treat risk outputs as review cues, not final decisions.
- Pair model outputs with manager context, workload context, promotion history, and business reality.
- Use department-level patterns to focus attention before jumping to individual conclusions.
- Read Methods & Limitations before presenting conclusions as if they were operational HR outputs.

## Common Questions

### Can this app make HR decisions automatically?

No. It is a review and discussion tool.

### Does the app retrain the model when I open it?

No. The app reads checked-in data, generated artifacts, static figures, and advanced reviewer metadata.

### Do I need an API key to use the app?

Not for the core pages. Some advanced PACE Navigator retrieval-backed reviewer features may require a local OpenAI API key.

The MLOps Lab optional AI briefing also needs `OPENAI_API_KEY` if you want generated summary text. Without a key, the deterministic CSV insight tables and packaged demo model scoring still work when their artifacts are present.

### Does OpenAI receive uploaded CSV rows?

No. Online CSV Insight sends only compact aggregate JSON for the optional briefing. Raw uploaded CSV rows and identifier-like fields are excluded.

### Does hosted Streamlit require FastAPI, Docker, MLflow, or Airflow?

No. Those are local/dev MLOps components for technical review. Hosted Streamlit uses the app entry point `app/app.py`, reads committed artifacts, and runs Online CSV Insight directly inside Streamlit.

### Is PACE the app name?

No. The app name is **Salifort Motors Retention Risk Explorer**. PACE is the workflow framing used in the advanced Navigator.

## Related Documents

- [Root README](../../README.md)
- [Documentation Guide](../README.md)
- [HR Quick Start](hr-quick-start.md)
- [Executive Summary](../executive/executive-summary.md)
- [Streamlit App Walkthrough](streamlit-app-walkthrough.md)
- [Technical Design and Architecture](../technical/technical-design-and-architecture.md)
- [Environment Setup and Deployment Guide](../deployment/environment-setup-and-deployment-guide.md)
