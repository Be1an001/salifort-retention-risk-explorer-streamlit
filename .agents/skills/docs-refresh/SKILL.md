---
name: docs-refresh
description: Refresh Salifort project documentation while preserving the existing docs architecture. Use when asked to update README, PRD, TDD, deployment guides, user manuals, formal docs, runbooks, or documentation indexes.
---

# Documentation Refresh Skill

## Inputs to Inspect

- `README.md`
- `docs/README.md`
- `docs/product/product-requirements-document.md`
- `docs/technical/technical-design-and-architecture.md`
- `docs/deployment/environment-setup-and-deployment-guide.md`
- `docs/user-guide/user-manual.md`
- `docs/user-guide/hr-quick-start.md`
- `docs/executive/executive-summary.md`
- `docs/formal/salifort-formal-document-package.md`
- MLOps and Navigator runbooks under `docs/`
- `app/app.py`
- `app/pages/mlops_lab.py`
- `artifacts/v2/metadata.json`
- `artifacts/mlops_lab_online/model_metadata.json`

## Steps

1. Audit existing docs before editing.
2. Identify stale, duplicated, missing, or misleading content.
3. Prefer small targeted updates over full rewrites.
4. Keep the root README concise and direct detailed readers to `docs/README.md`.
5. Keep canonical docs in their existing locations.
6. Add scope notes where documents overlap instead of deleting historical docs without a clear reason.
7. Preserve Traditional Chinese in business-facing docs when updating existing Chinese files.
8. Verify all model, MLOps, hosted, and responsible-use claims against code or artifacts.

## Boundaries

- Do not change Streamlit app behavior.
- Do not modify `artifacts/v2/`.
- Do not modify model artifacts.
- Do not claim production HR readiness, employment decision automation, hosted FastAPI, hosted Docker, hosted MLflow, or hosted Airflow.
- Do not claim OpenAI receives raw CSV rows or PII.
- Do not add AgentKit, Agents SDK, MCP, or production agent architecture for documentation-only tasks.

## Expected Outputs

- updated docs or indexes scoped to the requested documentation issue
- note of canonical documents and overlaps handled
- clear summary of files changed and validation run

## Done Criteria

- Docs reflect the nine-page app and current MLOps Lab behavior.
- Hosted Streamlit and local/dev MLOps responsibilities are separated.
- Public threshold `0.29` and lab threshold `0.60` are separated.
- Important docs are linked from `docs/README.md` when appropriate.
- No generated outputs or secrets are staged.
