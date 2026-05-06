# Product Requirements Document

## Document Purpose

This document defines the product framing for the **Salifort Motors Retention Risk Explorer**. It describes what the app is meant to do, who it is for, what is in scope today, and which boundaries should remain visible in the public portfolio version.

This is a portfolio and educational decision-support app. It is not a production HR platform and should not be used as an automated employment decision tool.

## Product Summary

The app packages an employee-retention-risk analytics project into a nine-page Streamlit experience that is easy to review, discuss, and demonstrate. It combines:

- a checked-in HR dataset
- generated model and explanation artifacts
- static project visuals
- optional advanced reviewer tools for evidence, citations, retrieval, workflow review, and readiness inspection
- a hosted MLOps Lab CSV sandbox with transparent heuristic scoring, packaged demo model inference, aggregate-only OpenAI briefing, and local/dev MLOps evidence

The product goal is not only to show model outputs, but to make the full review path easier to understand for different audiences.

## Core Business Question

How can Salifort Motors identify early retention risk, focus manager attention on the teams and groups that deserve review, and avoid treating a model score as an automated HR decision?

## Product Objectives

- Help a first-time visitor understand the business problem, model story, threshold choice, and review boundaries quickly.
- Support practical workforce review through interactive filtering, exposure summaries, and manager-oriented interpretation.
- Make model trade-offs and explainability visible without turning the app into a notebook dump.
- Preserve technical transparency through optional advanced reviewer tools that show evidence, citations, workflow readiness, and guarded plan previews.
- Keep offline build work and Streamlit runtime behavior clearly separated.

## Target Audiences

### 1. Recruiters, hiring managers, and interviewers

They need a clear portfolio story:

- what the project solves
- what pages are worth opening first
- what the model does and does not claim
- how the app demonstrates analytical, product, and communication skills

### 2. HR, HRBP, department managers, and manager-style readers

They need a practical review tool:

- department exposure and review priorities
- threshold trade-offs in business terms
- explanation of risk drivers without causal overclaiming
- clear responsible-use boundaries

### 3. Executives and senior stakeholders

They need a concise management story:

- what workforce risk pattern is being explored
- what value the app provides as a decision-support demo
- which risks require human review and governance
- why the app should not be treated as a production HR or employment-decision system

### 4. Technical reviewers

They need architecture and auditability:

- artifact-backed runtime design
- generated artifact contracts
- retrieval and citation behavior
- workflow/readiness boundaries
- hosted Streamlit model inference versus local/dev FastAPI, Docker, MLflow, Airflow, and CI boundaries

### 5. Developers and maintainers

They need runnable, inspectable project structure:

- environment setup and deployment steps
- app/runtime boundaries
- offline builders and validators
- clear documentation entry points

## In-Scope User Outcomes

The current product should help a visitor:

1. Understand the project question from the Overview page.
2. Explore workforce slices in the Workforce Explorer.
3. See the supporting data story in EDA & Patterns.
4. Understand model selection and threshold trade-offs in Model & Threshold Lab.
5. Review SHAP-based explanation in Explainability.
6. Translate the analysis into department review priorities in Manager Action View.
7. Understand architecture, limitations, and boundaries in Methods & Limitations.
8. Optionally inspect advanced evidence and review tooling in PACE Navigator.
9. Optionally review hosted CSV insight, packaged demo model inference, and local/dev MLOps evidence in MLOps Lab.

## Core Feature Set

### Overview

Acts as the landing page for the full project story, including:

- business question
- dataset and duplicate-removal context
- public model truth
- selected threshold truth
- suggested page order
- portfolio positioning and boundaries

### Workforce Explorer

Supports interactive review of:

- departments
- salary levels
- tenure bands
- selected-threshold or fallback risk flags
- department-level exposure summaries
- employee-level review rows

### EDA & Patterns

Shows stable visual evidence for:

- workload and satisfaction patterns
- department and salary structure
- tenure and project-load patterns

### Model & Threshold Lab

Explains:

- champion model comparison
- threshold trade-offs
- precision/recall context
- confusion-matrix behavior
- why threshold `0.29` is preserved as the public reference choice

### Explainability

Shows SHAP-based model interpretation and explains:

- which features drive the model signal
- how to interpret those drivers responsibly
- why SHAP describes model behavior rather than causality

### Manager Action View

Translates model outputs into practical review framing:

- total and normalized department exposure
- review-priority cues
- responsible manager interpretation

### Methods & Limitations

Explains:

- checked-in data and cleaning
- generated artifacts
- static figures
- offline build versus runtime behavior
- fallback logic
- retrieval and OpenAI API role
- PACE, workflow contracts, Airflow scaffold, and preview-only agent shell

### PACE Navigator

Provides optional advanced review tools for technical inspection:

- guided topic exploration
- source-of-truth and drift drilldowns
- fixed-question answer viewer
- citations and evidence inspection
- source preview eligibility
- multi-query audit workflow
- workflow contracts and readiness review
- preview-only plan shell

### MLOps Lab

Provides the ninth Streamlit page and separates hosted reviewer features from local/dev engineering evidence:

- Online CSV Insight for small Salifort-style uploads
- transparent pandas heuristic review scoring
- packaged MLOps Lab demo model inference from `artifacts/mlops_lab_online/`
- optional OpenAI briefing from compact aggregate JSON only
- static MLOps Evidence Pack snapshots
- local/dev command guidance for the MLOps pipeline, FastAPI, Docker Compose, MLflow, Airflow DAG validation, and CI

Important model boundary:

- the public app truth remains Weighted XGBoost at threshold `0.29`
- the MLOps Lab packaged demo model uses its own lab threshold, currently `0.60` in `artifacts/mlops_lab_online/model_metadata.json`
- the lab model and threshold do not replace the public app model story

## Representative User Stories

- As an HR reader, I want a safe way to inspect departments and risk bands so I can identify where human review should start.
- As a department manager, I want model outputs translated into review priorities so I can discuss workload, promotion, and team context before taking action.
- As an executive reviewer, I want a concise view of value, governance, and limitations so I can understand the demo without reading code.
- As a technical reviewer, I want to inspect architecture, artifacts, retrieval evidence, CI checks, and local/dev MLOps boundaries so I can evaluate engineering maturity.
- As a portfolio interviewer, I want a short guided path through the nine pages so I can discuss the project in a limited interview window.

## Non-Goals

The current repo does not aim to be:

- a production HR platform
- a live HR monitoring system
- an automated employment decision engine
- a free-form chatbot
- a production Airflow deployment
- an autonomous multi-agent system
- a hosted FastAPI, Docker, MLflow, or Airflow service
- a system that sends raw uploaded CSV rows or PII to OpenAI

## Success Criteria

### Visitor understanding

A first-time reviewer should be able to explain:

- the business question
- the public reference model
- why threshold `0.29` matters
- the difference between final model truth and fallback logic
- which pages to open first

### Portfolio usability

The app should support a short interview/demo path without requiring notebooks or local model training.

### Technical transparency

The advanced review layer should make sources, evidence, workflow contracts, and readiness boundaries inspectable without overwhelming the main app experience.

## Product Boundaries That Must Stay Explicit

- Streamlit reads local files; it does not retrain models during a visitor session.
- Fallback screening exists for resilience, but it is not the final model probability.
- Retrieval-backed review is fixed-question and evidence-oriented, not open-ended chat.
- OpenAI API use is limited to optional embedding-backed retrieval workflows and optional MLOps Lab aggregate-only briefings when the user supplies a key.
- Online CSV Insight sends compact aggregate JSON only for optional OpenAI briefings; raw uploaded CSV rows and identifier-like fields are excluded.
- Airflow and agent surfaces are review-oriented scaffolds and previews, not app-executed automation.
- FastAPI, Docker Compose, MLflow, and Airflow are local/dev MLOps components and do not run inside hosted Streamlit Community Cloud sessions.

## Future Opportunities

The repo leaves room for later expansion, such as:

- stronger intervention-tracking workflows
- richer threshold scenario simulation
- more realistic multi-period workforce monitoring
- deeper offline orchestration
- clearer reviewer packaging for screenshots and slide assets

Those are future directions, not current product commitments.
