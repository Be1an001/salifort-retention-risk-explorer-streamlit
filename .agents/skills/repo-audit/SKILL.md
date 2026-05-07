---
name: repo-audit
description: Audit this Salifort Streamlit repository for structure, evidence, portfolio positioning, and risk boundaries. Use when asked to inspect the repo, classify the project, prepare resume/interview material, or verify claims before making recommendations.
---

# Repo Audit Skill

## Inputs to Inspect

- `README.md`
- `docs/README.md`
- canonical docs under `docs/product/`, `docs/technical/`, `docs/deployment/`, `docs/user-guide/`, `docs/executive/`, and `docs/formal/`
- `app/app.py` and `app/pages/`
- `app/services/`, `app/viewmodels/`, and `app/utils/load_data.py`
- `artifacts/v2/metadata.json`
- `artifacts/mlops_lab_online/model_metadata.json`
- `navigator/truth_registry.json` and `navigator/drift_register.json`
- `requirements.txt`, `requirements-mlops.txt`, `docker-compose.yml`, `.github/workflows/ci.yml`
- relevant tests under `tests/`

## Steps

1. Inspect the current branch and worktree status.
2. Recursively inventory tracked project files and ignore dependency/cache/generated folders unless they affect the request.
3. Read the canonical docs before interpreting code behavior.
4. Verify app page count and entry point from `app/app.py`.
5. Verify public model truth from `artifacts/v2/metadata.json`.
6. Verify MLOps Lab packaged demo truth from `artifacts/mlops_lab_online/model_metadata.json`.
7. Separate hosted Streamlit behavior from local/dev MLOps behavior.
8. Identify confirmed evidence, unclear claims, drift, and overclaiming risks.
9. Summarize portfolio positioning using restrained, evidence-based wording.

## Boundaries

- Do not edit files during an audit unless the user explicitly asks for implementation.
- Do not infer production deployment, real HR use, business impact, or employment decision readiness.
- Do not attribute individual ownership unless confirmed from project files or user-provided context.
- Do not treat PACE Navigator as an autonomous agent system.

## Expected Outputs

- concise repository map
- confirmed app/model/MLOps facts
- classification or positioning if requested
- evidence paths for important claims
- risks, drift, and recommended next steps

## Done Criteria

- Key claims cite repository evidence.
- Unknowns are labeled as not confirmed.
- Public threshold `0.29` and lab threshold `0.60` are not conflated.
- The response preserves portfolio-demo and human-review boundaries.
