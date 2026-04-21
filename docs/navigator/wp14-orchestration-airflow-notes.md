# WP14 Orchestration And Airflow Notes

WP14 adds the first governed orchestration foundation for the PACE Navigator.
It defines workflow and task contracts, exposes read-only helper queries, and
adds an optional Airflow-ready scaffold. It does not add job execution from
Streamlit, production scheduling, agents, Airflow deployment, or new business
truth.

## Added Contracts

- `navigator/orchestration/task_registry.json` declares governed task units.
- `navigator/orchestration/workflow_registry.json` groups tasks into official
  workflows.
- Task contracts include inputs, outputs, dependencies, required artifacts,
  environment variables, blocked states, runtime mode, mutation behavior,
  scheduler eligibility, and human-review gates.

## Helper Layer

`app/services/navigator_orchestration.py` loads and validates the registries and
exposes read-only helpers such as:

- `list_workflows()`
- `list_tasks_for_workflow(workflow_id)`
- `get_task_definition(task_id)`
- `get_workflow_definition(workflow_id)`
- `get_execution_order(workflow_id)`
- `summarize_blockers(workflow_id)`
- `build_runbook_view(workflow_id)`

These helpers describe execution contracts. They do not execute jobs.

## Airflow Scaffold

`orchestration/airflow/dags/pace_navigator_governed_workflows.py` maps
Airflow-eligible workflow contracts into DAG skeletons when Airflow is installed.
If Airflow is not installed, the module imports safely and creates no DAG
objects.

This scaffold is for local/dev readiness only. It is not a production scheduler
configuration.

## Navigator Surface

The PACE Navigator shows an informational orchestration summary:

- governed workflow list
- workflow runtime and scheduler classes
- task order and dependencies
- artifact/env blockers
- mutation and human-review flags

No job execution buttons were added.

## Deferred

- Agent shell and governed planning.
- Production Airflow deployment.
- Background scheduling.
- Persistent workflow state.
- CI/CD redesign.
