from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.navigator_loader import get_repo_root
from app.services.navigator_types import NavigatorRegistryValidationError

_TASK_REQUIRED_FIELDS = {
    "task_id",
    "task_title",
    "task_kind",
    "stage",
    "phase",
    "inputs",
    "outputs",
    "dependencies",
    "required_artifacts",
    "required_env_vars",
    "blocked_states",
    "mutates_repo_files",
    "runtime_mode",
    "human_review_required",
    "scheduler_eligibility",
    "notes",
}
_WORKFLOW_REQUIRED_FIELDS = {
    "workflow_id",
    "workflow_title",
    "workflow_kind",
    "stage",
    "phase",
    "task_ids",
    "entry_task_id",
    "terminal_task_ids",
    "runtime_mode",
    "required_env_vars",
    "required_artifacts",
    "blocked_states",
    "scheduler_eligibility",
    "human_review_required",
    "mutates_repo_files",
    "notes",
}
_OPENAI_KEY_ALIASES = ("RAG_STREAMLIT_OPENAI_API_KEY", "OPENAI_API_KEY")


def get_orchestration_registry_paths() -> dict[str, Path]:
    root = get_repo_root() / "navigator" / "orchestration"
    return {
        "task_registry": root / "task_registry.json",
        "workflow_registry": root / "workflow_registry.json",
    }


def clear_orchestration_registry_caches() -> None:
    load_task_registry.cache_clear()
    load_workflow_registry.cache_clear()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise NavigatorRegistryValidationError(
            f"Required orchestration registry '{label}' is missing at {path}."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise NavigatorRegistryValidationError(
            f"Orchestration registry '{label}' is not valid JSON: {exc}"
        ) from exc


def _ensure_unique(records: list[dict[str, Any]], key: str, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        record_id = str(record.get(key, ""))
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)
    if duplicates:
        raise NavigatorRegistryValidationError(
            f"Duplicate {label} ids found: {sorted(duplicates)}"
        )


def _validate_task_registry(registry: dict[str, Any]) -> None:
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise NavigatorRegistryValidationError(
            "task_registry.json must contain a non-empty 'tasks' list."
        )
    _ensure_unique(tasks, "task_id", "task")
    task_ids = {task["task_id"] for task in tasks}
    for task in tasks:
        missing = _TASK_REQUIRED_FIELDS - set(task)
        if missing:
            raise NavigatorRegistryValidationError(
                f"Task {task.get('task_id', '<missing>')} is missing fields: {sorted(missing)}"
            )
        for list_field in (
            "inputs",
            "outputs",
            "dependencies",
            "required_artifacts",
            "required_env_vars",
            "blocked_states",
        ):
            if not isinstance(task[list_field], list):
                raise NavigatorRegistryValidationError(
                    f"Task {task['task_id']} field '{list_field}' must be a list."
                )
        for dependency in task["dependencies"]:
            if dependency not in task_ids:
                raise NavigatorRegistryValidationError(
                    f"Task {task['task_id']} depends on unknown task {dependency!r}."
                )


def _validate_workflow_registry(
    workflow_registry: dict[str, Any],
    task_registry: dict[str, Any],
) -> None:
    workflows = workflow_registry.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        raise NavigatorRegistryValidationError(
            "workflow_registry.json must contain a non-empty 'workflows' list."
        )
    _ensure_unique(workflows, "workflow_id", "workflow")
    task_ids = {task["task_id"] for task in task_registry["tasks"]}
    task_lookup = {task["task_id"]: task for task in task_registry["tasks"]}
    for workflow in workflows:
        missing = _WORKFLOW_REQUIRED_FIELDS - set(workflow)
        if missing:
            raise NavigatorRegistryValidationError(
                f"Workflow {workflow.get('workflow_id', '<missing>')} is missing fields: {sorted(missing)}"
            )
        for list_field in (
            "task_ids",
            "terminal_task_ids",
            "required_env_vars",
            "required_artifacts",
            "blocked_states",
        ):
            if not isinstance(workflow[list_field], list):
                raise NavigatorRegistryValidationError(
                    f"Workflow {workflow['workflow_id']} field '{list_field}' must be a list."
                )
        for task_id in workflow["task_ids"]:
            if task_id not in task_ids:
                raise NavigatorRegistryValidationError(
                    f"Workflow {workflow['workflow_id']} references unknown task {task_id!r}."
                )
        if workflow["entry_task_id"] not in workflow["task_ids"]:
            raise NavigatorRegistryValidationError(
                f"Workflow {workflow['workflow_id']} entry task is not in task_ids."
            )
        for terminal_task_id in workflow["terminal_task_ids"]:
            if terminal_task_id not in workflow["task_ids"]:
                raise NavigatorRegistryValidationError(
                    f"Workflow {workflow['workflow_id']} terminal task is not in task_ids."
                )
        positions = {
            task_id: index for index, task_id in enumerate(workflow["task_ids"])
        }
        for task_id in workflow["task_ids"]:
            for dependency in task_lookup[task_id]["dependencies"]:
                if dependency in positions and positions[dependency] > positions[task_id]:
                    raise NavigatorRegistryValidationError(
                        f"Workflow {workflow['workflow_id']} orders {task_id!r} before dependency {dependency!r}."
                    )


@lru_cache(maxsize=1)
def load_task_registry() -> dict[str, Any]:
    registry = _load_json(
        get_orchestration_registry_paths()["task_registry"],
        "task_registry",
    )
    _validate_task_registry(registry)
    return registry


@lru_cache(maxsize=1)
def load_workflow_registry() -> dict[str, Any]:
    task_registry = load_task_registry()
    registry = _load_json(
        get_orchestration_registry_paths()["workflow_registry"],
        "workflow_registry",
    )
    _validate_workflow_registry(registry, task_registry)
    return registry


def load_orchestration_registries() -> dict[str, Any]:
    return {
        "task_registry": load_task_registry(),
        "workflow_registry": load_workflow_registry(),
    }


def _task_lookup() -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in load_task_registry()["tasks"]}


def _workflow_lookup() -> dict[str, dict[str, Any]]:
    return {
        workflow["workflow_id"]: workflow
        for workflow in load_workflow_registry()["workflows"]
    }


def list_workflows() -> list[dict[str, Any]]:
    return sorted(
        load_workflow_registry()["workflows"],
        key=lambda workflow: workflow["workflow_id"],
    )


def get_workflow_definition(workflow_id: str) -> dict[str, Any]:
    workflow = _workflow_lookup().get(workflow_id)
    if workflow is None:
        raise KeyError(f"Unknown workflow_id {workflow_id!r}.")
    return workflow


def get_task_definition(task_id: str) -> dict[str, Any]:
    task = _task_lookup().get(task_id)
    if task is None:
        raise KeyError(f"Unknown task_id {task_id!r}.")
    return task


def list_tasks_for_workflow(workflow_id: str) -> list[dict[str, Any]]:
    workflow = get_workflow_definition(workflow_id)
    task_lookup = _task_lookup()
    return [task_lookup[task_id] for task_id in workflow["task_ids"]]


def get_execution_order(workflow_id: str) -> list[str]:
    # Workflow registry task_ids are validated as dependency-safe order.
    return list(get_workflow_definition(workflow_id)["task_ids"])


def _env_requirement_status(required_env_vars: list[str]) -> dict[str, Any]:
    if not required_env_vars:
        return {
            "status": "not_required",
            "label": "No environment variables required",
            "missing": [],
        }
    if all(alias in required_env_vars for alias in _OPENAI_KEY_ALIASES):
        key_available = any(os.environ.get(alias) for alias in _OPENAI_KEY_ALIASES)
        return {
            "status": "satisfied" if key_available else "blocked",
            "label": "OpenAI API key available via approved environment alias"
            if key_available
            else "OpenAI API key missing; set RAG_STREAMLIT_OPENAI_API_KEY or OPENAI_API_KEY locally",
            "missing": [] if key_available else ["RAG_STREAMLIT_OPENAI_API_KEY or OPENAI_API_KEY"],
        }
    missing = [name for name in required_env_vars if not os.environ.get(name)]
    return {
        "status": "satisfied" if not missing else "blocked",
        "label": "Required environment variables are present"
        if not missing
        else "One or more required environment variables are missing",
        "missing": missing,
    }


def _artifact_requirement_status(required_artifacts: list[str]) -> dict[str, Any]:
    repo_root = get_repo_root()
    missing = [
        artifact
        for artifact in required_artifacts
        if not (repo_root / artifact).exists()
    ]
    return {
        "status": "satisfied" if not missing else "blocked",
        "missing": missing,
    }


def summarize_blockers(workflow_id: str) -> dict[str, Any]:
    workflow = get_workflow_definition(workflow_id)
    tasks = list_tasks_for_workflow(workflow_id)
    required_env_vars = sorted(
        {
            env_var
            for item in [workflow, *tasks]
            for env_var in item["required_env_vars"]
        }
    )
    required_artifacts = sorted(
        {
            artifact
            for item in [workflow, *tasks]
            for artifact in item["required_artifacts"]
        }
    )
    env_status = _env_requirement_status(required_env_vars)
    artifact_status = _artifact_requirement_status(required_artifacts)
    blocker_notes = list(workflow["blocked_states"])
    for task in tasks:
        blocker_notes.extend(task["blocked_states"])
    blocker_notes = sorted(set(blocker_notes))
    status = (
        "blocked"
        if env_status["status"] == "blocked" or artifact_status["status"] == "blocked"
        else "ready_for_contract_review"
    )
    return {
        "workflow_id": workflow_id,
        "status": status,
        "required_env_vars": required_env_vars,
        "env_status": env_status,
        "required_artifacts": required_artifacts,
        "artifact_status": artifact_status,
        "blocked_states": blocker_notes,
    }


def build_runbook_view(workflow_id: str) -> dict[str, Any]:
    workflow = get_workflow_definition(workflow_id)
    tasks = list_tasks_for_workflow(workflow_id)
    blockers = summarize_blockers(workflow_id)
    return {
        "workflow": workflow,
        "execution_order": get_execution_order(workflow_id),
        "tasks": tasks,
        "blockers": blockers,
        "summary": {
            "workflow_id": workflow["workflow_id"],
            "workflow_title": workflow["workflow_title"],
            "workflow_kind": workflow["workflow_kind"],
            "runtime_mode": workflow["runtime_mode"],
            "scheduler_eligibility": workflow["scheduler_eligibility"],
            "human_review_required": workflow["human_review_required"],
            "mutates_repo_files": workflow["mutates_repo_files"],
            "task_count": len(tasks),
            "blocked_state_count": len(blockers["blocked_states"]),
        },
    }
