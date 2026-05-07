---
name: responsible-ai-review
description: Review Salifort AI briefing, Online CSV Insight, PACE Navigator, and responsible-use behavior. Use when asked to evaluate OpenAI privacy boundaries, no-PII handling, employment-decision wording, threshold separation, or eval cases under evals/.
---

# Responsible AI Review Skill

## Inputs to Inspect

- `app/pages/mlops_lab.py`
- `app/pages/methods_limitations.py`
- `app/pages/pace_navigator.py`
- `tests/test_online_csv_insight_contract.py`
- `tests/test_mlops_evidence_pack_contract.py`
- `evals/`
- `docs/user-guide/user-manual.md`
- `docs/user-guide/hr-quick-start.md`
- `docs/executive/executive-summary.md`
- `artifacts/v2/metadata.json`
- `artifacts/mlops_lab_online/model_metadata.json`

## Steps

1. Identify whether the review concerns heuristic scoring, packaged model scoring, public app model truth, PACE Navigator, or OpenAI briefing.
2. Verify that AI briefing inputs are compact aggregate JSON, not raw uploaded rows.
3. Verify PII-like fields are excluded from briefing payloads and downloads.
4. Verify output language avoids employment decisions, automated action, production HR claims, and causal overclaims.
5. Verify threshold language separates public `0.29` from lab `0.60`.
6. Use `evals/*.jsonl` cases as lightweight behavior checks when drafting or reviewing briefing prompts and responsible-use text.
7. Recommend human-review wording for ambiguous cases.

## Boundaries

- Do not add new OpenAI runtime behavior unless explicitly requested.
- Do not add AgentKit, Agents SDK, MCP, or production guardrail services.
- Do not send real data or sample rows to external services during review.
- Do not present lightweight JSONL cases as a full production eval platform.
- Do not soften responsible-use boundaries to make the app sound more automated.

## Expected Outputs

- pass/fail or risk summary for responsible-use behavior
- suggested wording fixes when needed
- eval case updates only when requested or necessary for the task
- validation commands run or skipped with rationale

## Done Criteria

- No raw CSV row, PII, or employment-decision language is introduced.
- Public and lab thresholds remain distinct.
- Any eval case additions are valid JSONL and contain no real PII.
