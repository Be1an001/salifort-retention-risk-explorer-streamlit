---
name: streamlit-page-review
description: Review or modify Salifort Streamlit pages while preserving page count, artifact-backed behavior, and responsible-use copy. Use when asked to inspect, polish, or validate app pages under app/app.py and app/pages/.
---

# Streamlit Page Review Skill

## Inputs to Inspect

- `app/app.py`
- target files under `app/pages/`
- `app/utils/load_data.py`
- relevant `app/services/` and `app/viewmodels/` files
- `artifacts/v2/metadata.json`
- `artifacts/mlops_lab_online/model_metadata.json`
- relevant tests under `tests/`
- relevant docs for page wording boundaries

## Steps

1. Confirm the page route and navigation entry in `app/app.py`.
2. Read the target page and helper functions before proposing edits.
3. Identify whether the page consumes public app artifacts, MLOps Lab artifacts, Navigator metadata, or static figures.
4. Preserve read-only runtime behavior unless the user explicitly requests a functional change.
5. Keep user-facing copy clear about human review, portfolio-demo status, and threshold/model boundaries.
6. Prefer small, local edits consistent with existing page style.
7. Run py_compile and focused tests when page behavior or contract wording changes.

## Boundaries

- Do not run Streamlit unless the user asks.
- Do not run training, export scripts, FastAPI, MLflow, Docker services, or Airflow.
- Do not send uploaded data to external services.
- Do not introduce required hosted FastAPI configuration.
- Do not change OpenAI call behavior unless explicitly requested.
- Do not turn PACE Navigator into a free-form chatbot or autonomous runner.

## Expected Outputs

- page-level findings or scoped page edits
- explanation of affected route, artifacts, and validation
- any unresolved UX, data, or safety caveats

## Done Criteria

- The nine-page app structure remains intact unless the user requested a page change.
- App logic remains artifact-backed and read-only for visitor sessions.
- Responsible-use and OpenAI privacy boundaries remain visible.
- Focused validations pass or failures are clearly reported.
