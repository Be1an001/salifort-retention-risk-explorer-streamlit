# Streamlit App Walkthrough

Use this walkthrough when reviewing the **Salifort Motors Retention Risk Explorer** for the first time.

## Recommended Order

1. Overview
2. Workforce Explorer
3. EDA & Patterns
4. Model & Threshold Lab
5. Explainability
6. Manager Action View
7. Methods & Limitations
8. PACE Navigator for optional advanced review
9. MLOps Lab for optional technical operations review

## 1. Overview

Start here to understand the project in one page.

What to look for:

- the business question
- cleaned dataset size and duplicate removal
- public reference model: weighted XGBoost
- selected threshold: `0.29`
- what the app does and does not do
- suggested page order

How to interpret it:

- the metrics are portfolio context, not live HR monitoring
- the app supports responsible review, not automated employment decisions
- the advanced reviewer layer is optional

## 2. Workforce Explorer

Use the controls to filter by:

- department
- salary
- tenure band
- selected-threshold or fallback high-risk flag

What to look for:

- how filtered headcount changes
- which departments have higher total exposure
- whether observed attrition, predicted attrition, and flagged counts point in the same direction
- which employee-level fields sit behind the summary

How to interpret it:

- generated row-level artifacts show model-backed fields when available
- fallback screening keeps the page explorable when generated files are absent
- fallback scores are not final model probabilities

## 3. EDA & Patterns

Use the tabs to review stable project visuals:

- **Workload Patterns**
- **Department / Salary Structure**
- **Tenure / Project Structure**

What to look for:

- clusters where workload and satisfaction look concerning
- departments or salary bands where attrition patterns differ
- tenure/project combinations that may show sustained pressure

## 4. Model & Threshold Lab

Use this page to understand why the threshold matters.

What to inspect:

- champion model and headline metrics
- model comparison table or figure
- precision-recall chart
- threshold trade-off curve
- confusion matrix at the selected threshold

How to interpret it:

- accuracy alone is not enough when attrition is the smaller outcome
- lower thresholds can catch more likely leavers but create more false positives
- threshold choice is a business-review trade-off, not just a modeling detail

## 5. Explainability

Use this page to see which features influence the model signal.

What to look for:

- ranked SHAP importance table or chart
- feature-driver descriptions written for portfolio review
- reference SHAP visuals if generated tables are unavailable

How to interpret it:

- SHAP explains the model's behavior
- SHAP does not prove what caused attrition
- use this page to support discussion, not to make causal claims

## 6. Manager Action View

Use this page to translate analysis into responsible review priorities.

What to look for:

- departments with high total exposure
- departments with high exposure per 100 employees
- the relationship between review volume and threshold choice

How to interpret it:

- start with departments that show both scale and intensity
- pair model signals with workload, promotion, team context, and local review
- do not penalize or rank employees based only on this app

## 7. Methods & Limitations

Use this page to understand how the app is built.

Key ideas:

- the checked-in CSV keeps the demo reproducible
- generated artifacts hold model outputs, threshold tables, exposure tables, and SHAP summaries
- static figures preserve the visual project story
- Streamlit reads local files and renders pages
- model training and artifact generation happen offline
- fallback logic keeps the app usable but stays separate from final model truth

This page also explains advanced terms such as PACE, retrieval/RAG, Airflow scaffold, and agent shell.

## 8. PACE Navigator

Use this page after the main app pages, especially if you are doing a technical review.

Beginner-friendly areas:

- current public model truth
- guided topic explorer
- suggested next pages
- PACE workflow map

Advanced review tools:

- fixed-question answer viewer with retrieval depth control
- citations, source detail, and retrieval inspector
- source preview eligibility
- multi-query audit export
- workflow contracts and readiness
- controlled plan-preview shell

How to interpret retrieval features:

- retrieval uses prepared project chunks, not arbitrary web search
- OpenAI embeddings support retrieval when the reviewer chooses a fixed question and retrieval depth
- the answer viewer assembles a structured answer with evidence, citations, caveats, and coverage notes
- it is not a chatbot and does not execute jobs

## 9. MLOps Lab

Use this page only if you want to review the optional MLOps Mini-Lab extension.

What to inspect:

- package and CLI pipeline status
- generated local lab artifact availability
- lab champion and MLflow tracking summary when local reports exist
- optional FastAPI health and model-info status
- Docker Compose, Airflow DAG, and CI validation surfaces

How to interpret it:

- the page is read-only and does not run training, Docker, MLflow, Airflow, git, or CI commands
- generated lab artifacts are optional, local/dev, and gitignored
- the lab model and lab threshold do not replace the public weighted XGBoost threshold `0.29` story
- this is technical portfolio review support, not production HR infrastructure

## Static, Artifact-Backed, and Retrieval-Backed Outputs

### Static outputs

Mainly the PNG figures in `outputs/figures/`.

### Artifact-backed outputs

Mainly the generated runtime artifacts in `artifacts/v2/`, including model metadata, threshold tables, exposure summaries, row-level scores, and SHAP summaries.

### Retrieval-backed outputs

Only the advanced PACE Navigator review tools that use the governed retrieval pack and embedding index. These may require a local OpenAI API key.

## Short Interview Path

For a short interview walkthrough:

1. Show Overview for project framing.
2. Use Workforce Explorer filters to show interactivity.
3. Open Model & Threshold Lab to discuss trade-offs.
4. Open Explainability to discuss SHAP responsibly.
5. Open Manager Action View to show practical review framing.
6. Open Methods & Limitations to show architecture and boundaries.
7. Open PACE Navigator only if the reviewer wants advanced audit and retrieval details.
8. Open MLOps Lab only if the reviewer wants local/dev MLOps implementation details.

## Related Documents

- [User Manual](user-manual.md)
- [Technical Design and Architecture](../technical/technical-design-and-architecture.md)
- [Environment Setup and Deployment Guide](../deployment/environment-setup-and-deployment-guide.md)
- [Navigator Notes](../navigator/README.md)
