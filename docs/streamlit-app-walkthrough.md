# Streamlit App Walkthrough

Use this walkthrough when reviewing the Salifort Motors Retention Risk Explorer for the first time.

## 1. Start With Overview

Open **Overview** first. This page explains the project question, shows the cleaned dataset size, and summarizes the public reference model and threshold.

What to look for:

- The project is a portfolio demo, not a production HR system.
- The public reference model is weighted XGBoost.
- The selected threshold is `0.29`.
- The app supports responsible review rather than automated employment action.

## 2. Explore Workforce Patterns

Open **Workforce Explorer** to filter the cleaned HR dataset by department, salary, tenure band, and risk flag.

Use this page to answer:

- Which departments have higher exposure?
- How do headcount, observed attrition, and predicted risk change after filtering?
- Which employee-level fields are visible for review?

If generated row-level artifacts are unavailable, this page may use a simpler fallback screening score. The app labels that fallback clearly.

## 3. Review EDA Figures

Open **EDA & Patterns** for the stable visual summary of the analysis workflow.

Useful tabs:

- **Workload Patterns**: hours, satisfaction, salary, and retention signals.
- **Department / Salary Structure**: how department, salary, promotion, and attrition patterns overlap.
- **Tenure / Project Structure**: where project load and tenure combine.

## 4. Compare Model and Threshold Trade-Offs

Open **Model & Threshold Lab** to review model comparison, precision-recall behavior, and the selected threshold.

This page is useful for interview discussion because it shows that the project does not treat a model score as enough by itself. The threshold affects recall, precision, and the size of the review queue.

## 5. Read the Explainability Page

Open **Explainability** to see SHAP-based model interpretation.

Use this page to explain which features most influence the model's risk signal. Keep the limitation clear: SHAP explains the model; it does not prove what caused attrition.

## 6. Translate Findings Into Review Priorities

Open **Manager Action View** to see department exposure and practical review guidance.

Use this page to discuss where manager attention might start, while keeping human review and local context in the loop.

## 7. Check Methods and Limits

Open **Methods & Limitations** to understand the runtime design.

Key points:

- Streamlit reads existing files.
- Streamlit does not retrain models.
- Offline scripts build generated artifacts.
- The fallback screening score is separate from final model probability.
- This is an educational portfolio project.

## 8. Use PACE Navigator for Advanced Review

Open **PACE Navigator** after the main portfolio pages.

This page is the advanced technical review area. It includes:

- Project topic routing.
- Fixed-question answer viewer.
- Citations and retrieval inspector.
- Source-preview eligibility.
- Multi-query audit export.
- Workflow contracts and demo readiness.
- A controlled plan-preview shell.

The Navigator is intentionally not a chatbot and does not execute workflows. It is there to make the project more auditable for technical reviewers.

## Suggested Interview Path

For a short review, use this order:

1. Overview
2. Workforce Explorer
3. Model & Threshold Lab
4. Explainability
5. Manager Action View
6. Methods & Limitations
7. PACE Navigator, only if the reviewer wants the advanced audit layer
