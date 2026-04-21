# Streamlit App Walkthrough

Use this walkthrough when reviewing the Salifort Motors Retention Risk Explorer for the first time.

The recommended path is:

1. Overview
2. Workforce Explorer
3. EDA & Patterns
4. Model & Threshold Lab
5. Explainability
6. Manager Action View
7. Methods & Limitations
8. PACE Navigator for optional advanced review

## 1. Overview

Start here to understand the project in one page.

What to look for:

- The project question.
- Cleaned dataset size and duplicate removal.
- Public reference model: weighted XGBoost.
- Selected threshold: `0.29`.
- What the app does and does not do.
- Suggested page order for review.

How to interpret it:

- The metrics are portfolio context, not live HR monitoring.
- The portfolio summary figure is a presentation artifact.
- The app supports responsible review, not automated employment decisions.

## 2. Workforce Explorer

Use the sidebar controls to filter:

- Department
- Salary
- Tenure band
- Selected-threshold or fallback risk flag

What to look for:

- How filtered headcount changes.
- Which departments have higher total exposure.
- Whether observed attrition, predicted attrition, and flagged counts point in the same direction.
- Which employee-level fields sit behind the summary.

How to interpret it:

- Generated row-level artifacts show model-backed fields when available.
- Fallback screening keeps the page explorable when generated files are absent.
- Fallback scores are not final model probabilities.

## 3. EDA & Patterns

Use the tabs to review stable project visuals:

- **Workload Patterns:** hours, satisfaction, salary, and retention signals.
- **Department / Salary Structure:** how department, salary, promotion, and attrition overlap.
- **Tenure / Project Structure:** how project load and tenure combine.

What to look for:

- Clusters where workload and satisfaction look concerning.
- Departments or salary bands where attrition patterns differ.
- Tenure/project combinations that may show sustained pressure.

## 4. Model & Threshold Lab

Use this page to understand why the threshold matters.

What to inspect:

- Champion model and headline metrics.
- Model comparison table or figure.
- Precision-recall chart.
- Threshold trade-off curve.
- Confusion matrix at the selected threshold.

How to interpret it:

- Accuracy alone is not enough when attrition is the smaller outcome.
- Lower thresholds can catch more likely leavers but create more false positives.
- Threshold choice is a business-review trade-off, not just a modeling detail.

## 5. Explainability

Use this page to see which features influence the model signal.

What to look for:

- Ranked SHAP importance table or chart.
- Plain-English driver descriptions.
- Reference SHAP visuals if generated tables are unavailable.

How to interpret it:

- SHAP explains the model's behavior.
- SHAP does not prove what caused attrition.
- Use this page to support discussion, not to make causal claims.

## 6. Manager Action View

Use this page to translate analysis into responsible review priorities.

What to look for:

- Departments with high total exposure.
- Departments with high exposure per 100 employees.
- The relationship between review volume and threshold choice.

How to interpret it:

- Start with departments that show both scale and intensity.
- Pair model signals with workload, promotion, team context, and local review.
- Do not penalize or rank employees based only on this app.

## 7. Methods & Limitations

Use this page to understand how the app is built.

Key ideas:

- The checked-in CSV keeps the demo reproducible.
- Generated artifacts hold model outputs, threshold tables, exposure tables, and SHAP summaries.
- Static figures preserve the visual project story.
- Streamlit reads local files and renders pages.
- Model training and artifact generation happen offline.
- Fallback logic keeps the app usable but stays separate from final model truth.

This page also explains advanced terms such as PACE, retrieval/RAG, Airflow scaffold, and agent shell in simple English.

## 8. PACE Navigator

Use this page after the main app pages, especially if you are doing a technical review.

Beginner-friendly areas:

- Current public model truth.
- Guided topic explorer.
- Suggested next pages.
- PACE workflow map.

Advanced review tools:

- Fixed-question answer viewer.
- Retrieval depth slider for prepared evidence chunks.
- Citations, source detail, and retrieval inspector.
- Source preview eligibility.
- Multi-query audit export.
- Workflow contracts and demo readiness.
- Controlled plan-preview shell.

How to interpret retrieval features:

- Retrieval uses prepared project chunks, not arbitrary web search.
- API-backed embedding calls require a local user-provided key.
- The answer viewer uses fixed questions and citation-backed assembly.
- It is not a chatbot and does not execute jobs.

## Short Interview Path

For a short interview walkthrough:

1. Show Overview for the project framing.
2. Use Workforce Explorer filters to show interactivity.
3. Open Model & Threshold Lab to discuss trade-offs.
4. Open Explainability to discuss SHAP responsibly.
5. Open Manager Action View to show practical review framing.
6. Open Methods & Limitations to show architecture and boundaries.
7. Open PACE Navigator only if the reviewer wants advanced audit/retrieval details.
