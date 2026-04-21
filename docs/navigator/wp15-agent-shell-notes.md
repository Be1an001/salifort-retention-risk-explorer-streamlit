# WP15 Governed PACE Agent Shell Notes

WP15 adds the first governed PACE agent shell. It is a controlled planner and
route preview layer, not an autonomous agent and not a job executor.

## What Was Added

- `navigator/agent/agent_policy.json` defines supported intent classes,
  disallowed behaviors, and preview-only execution policy.
- `navigator/agent/request_catalog.json` defines controlled request scenarios.
- `app/services/navigator_agent_shell.py` validates policy/catalog mappings and
  builds read-only plan previews from orchestration contracts.
- The PACE Navigator includes a controlled agent-shell section with request
  selection, intent routing, workflow/task mapping, blockers, required inputs,
  expected outputs, review checkpoints, and guardrails.

## Governance Boundaries

- No workflows are executed.
- No Airflow DAG is triggered.
- No free-text agent prompt box is added.
- No background jobs, persistence, or agent memory are added.
- No secret value is read, printed, transformed, or stored.
- All supported requests must map to existing orchestration workflow/task
  contracts.

## Supported Direction

The shell can preview:

- truth inspection routes
- drift inspection routes
- workflow/blocker inspection routes
- retrieval-quality review routes
- answer-support review routes
- export-preparation routes
- source-preview eligibility routes

## Deferred

- Actual governed execution approvals.
- Agent action logs or persistence.
- Airflow trigger integration.
- Free-form natural-language request interpretation.
- Multi-step autonomous planning.
