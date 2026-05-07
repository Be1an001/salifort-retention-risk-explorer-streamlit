# Lightweight Responsible-Use Eval Cases

This folder contains lightweight repository evaluation cases for the Salifort Motors Retention Risk Explorer. These files are not a full production eval platform, do not run automatically, and do not introduce a production agent architecture.

The cases document expected behavior for Online CSV Insight AI briefings, responsible-use language, PII boundaries, and model-threshold separation. They are intended for maintainers and coding agents to inspect when changing prompts, documentation, tests, or MLOps Lab copy.

## Files

- `online_csv_briefing_cases.jsonl`: cases for optional OpenAI aggregate briefings, heuristic scoring, packaged demo model scoring, and aggregate-only payload expectations.
- `responsible_use_cases.jsonl`: cases for human-review wording, no employment-decision claims, no production HR claims, and PACE Navigator / MLOps Lab boundaries.
- `pii_safety_cases.jsonl`: cases for PII-like column handling, no raw row leakage, and safe aggregate summaries.

## Case Format

Each file uses JSONL: one JSON object per line. Cases use simple fields so they can be inspected or adapted later:

- `id`
- `category`
- `input_summary`
- `expected_behavior`
- `forbidden_behavior`
- `pass_criteria`
- `source_refs`

The cases intentionally avoid real PII and raw employee-level records. Use aggregate descriptions, schema summaries, and boundary checks instead of row-level data.

## How to Extend

Add a new JSON object on one line. Keep cases small, specific, and tied to current repo behavior. Prefer expectations that can be checked by reading output text or serialized aggregate payloads.

When extending cases:

- preserve the public app truth: weighted XGBoost at threshold `0.29`
- preserve the MLOps Lab packaged demo threshold: `0.60`
- keep heuristic scoring distinct from packaged model inference
- require human-review language
- forbid employment decision, production HR, and causal claims
- forbid raw CSV rows and PII in OpenAI-bound payloads
- cite relevant repo files in `source_refs`
