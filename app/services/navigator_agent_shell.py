from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.navigator_loader import get_repo_root
from app.services.navigator_orchestration import (
    build_runbook_view,
    get_task_definition,
    get_workflow_definition,
    summarize_blockers,
)
from app.services.navigator_types import NavigatorRegistryValidationError

_POLICY_REQUIRED_KEYS = {
    "policy_version",
    "repo_name",
    "scope",
    "execution_policy",
    "supported_intents",
    "disallowed_behaviors",
}
_INTENT_REQUIRED_KEYS = {
    "intent_id",
    "intent_title",
    "description",
    "allowed_operations",
    "workflow_ids",
    "task_ids",
    "page_routes",
    "pace_phase",
    "human_review_required",
    "api_dependency_mode",
}
_REQUEST_REQUIRED_KEYS = {
    "request_id",
    "label",
    "controlled_request",
    "intent_id",
    "request_class",
    "pace_phase",
    "workflow_id",
    "task_ids",
    "page_routes",
    "governed_query",
    "required_inputs",
    "expected_outputs",
    "review_checkpoints",
    "notes",
}
_DISALLOWED_TERMS = {
    "arbitrary_repo_execution": ("run every workflow", "execute workflow", "modify files"),
    "unconstrained_chat": ("ask anything", "ask the agent", "free text", "general assistant"),
    "hidden_chain_execution": ("trigger airflow", "background", "silently"),
    "secret_handling": ("api key", "token", "secret", ".env"),
    "workflow_mutation": ("invent a workflow", "rewrite contracts", "add tasks"),
}


def get_agent_registry_paths() -> dict[str, Path]:
    root = get_repo_root() / "navigator" / "agent"
    return {
        "agent_policy": root / "agent_policy.json",
        "request_catalog": root / "request_catalog.json",
    }


def clear_agent_shell_caches() -> None:
    load_agent_policy.cache_clear()
    load_request_catalog.cache_clear()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise NavigatorRegistryValidationError(
            f"Required agent shell registry '{label}' is missing at {path}."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise NavigatorRegistryValidationError(
            f"Agent shell registry '{label}' is not valid JSON: {exc}"
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


def _validate_policy(policy: dict[str, Any]) -> None:
    missing = _POLICY_REQUIRED_KEYS - set(policy)
    if missing:
        raise NavigatorRegistryValidationError(
            f"agent_policy.json is missing fields: {sorted(missing)}"
        )
    execution_policy = policy["execution_policy"]
    if execution_policy.get("autonomous_execution_allowed") is not False:
        raise NavigatorRegistryValidationError(
            "Agent policy must explicitly disallow autonomous execution."
        )
    intents = policy["supported_intents"]
    if not isinstance(intents, list) or not intents:
        raise NavigatorRegistryValidationError(
            "agent_policy.json must contain supported intents."
        )
    _ensure_unique(intents, "intent_id", "intent")
    for intent in intents:
        missing = _INTENT_REQUIRED_KEYS - set(intent)
        if missing:
            raise NavigatorRegistryValidationError(
                f"Intent {intent.get('intent_id', '<missing>')} is missing fields: {sorted(missing)}"
            )
        for list_field in ("allowed_operations", "workflow_ids", "task_ids", "page_routes"):
            if not isinstance(intent[list_field], list):
                raise NavigatorRegistryValidationError(
                    f"Intent {intent['intent_id']} field '{list_field}' must be a list."
                )
        for workflow_id in intent["workflow_ids"]:
            get_workflow_definition(workflow_id)
        for task_id in intent["task_ids"]:
            get_task_definition(task_id)

    disallowed = policy["disallowed_behaviors"]
    if not isinstance(disallowed, list) or not disallowed:
        raise NavigatorRegistryValidationError(
            "agent_policy.json must contain disallowed behaviors."
        )
    _ensure_unique(disallowed, "behavior_id", "disallowed behavior")


def _validate_catalog(catalog: dict[str, Any], policy: dict[str, Any]) -> None:
    requests = catalog.get("requests")
    if not isinstance(requests, list) or not requests:
        raise NavigatorRegistryValidationError(
            "request_catalog.json must contain governed requests."
        )
    intent_ids = {intent["intent_id"] for intent in policy["supported_intents"]}
    _ensure_unique(requests, "request_id", "request")
    for request in requests:
        missing = _REQUEST_REQUIRED_KEYS - set(request)
        if missing:
            raise NavigatorRegistryValidationError(
                f"Request {request.get('request_id', '<missing>')} is missing fields: {sorted(missing)}"
            )
        if request["intent_id"] not in intent_ids:
            raise NavigatorRegistryValidationError(
                f"Request {request['request_id']} references unknown intent {request['intent_id']!r}."
            )
        get_workflow_definition(request["workflow_id"])
        workflow_tasks = {
            task["task_id"]
            for task in build_runbook_view(request["workflow_id"])["tasks"]
        }
        for task_id in request["task_ids"]:
            get_task_definition(task_id)
            if task_id not in workflow_tasks:
                raise NavigatorRegistryValidationError(
                    f"Request {request['request_id']} maps task {task_id!r} outside workflow {request['workflow_id']!r}."
                )
        for list_field in (
            "task_ids",
            "page_routes",
            "required_inputs",
            "expected_outputs",
            "review_checkpoints",
        ):
            if not isinstance(request[list_field], list):
                raise NavigatorRegistryValidationError(
                    f"Request {request['request_id']} field '{list_field}' must be a list."
                )


@lru_cache(maxsize=1)
def load_agent_policy() -> dict[str, Any]:
    policy = _load_json(get_agent_registry_paths()["agent_policy"], "agent_policy")
    _validate_policy(policy)
    return policy


@lru_cache(maxsize=1)
def load_request_catalog() -> dict[str, Any]:
    policy = load_agent_policy()
    catalog = _load_json(
        get_agent_registry_paths()["request_catalog"],
        "request_catalog",
    )
    _validate_catalog(catalog, policy)
    return catalog


def load_agent_shell_registries() -> dict[str, Any]:
    return {
        "agent_policy": load_agent_policy(),
        "request_catalog": load_request_catalog(),
    }


def get_supported_agent_intents() -> list[dict[str, Any]]:
    return sorted(
        load_agent_policy()["supported_intents"],
        key=lambda item: item["intent_id"],
    )


def get_disallowed_agent_behaviors() -> list[dict[str, Any]]:
    return sorted(
        load_agent_policy()["disallowed_behaviors"],
        key=lambda item: item["behavior_id"],
    )


def get_controlled_agent_requests() -> list[dict[str, Any]]:
    return sorted(
        load_request_catalog()["requests"],
        key=lambda item: item["request_id"],
    )


def _intent_lookup() -> dict[str, dict[str, Any]]:
    return {intent["intent_id"]: intent for intent in load_agent_policy()["supported_intents"]}


def _request_lookup() -> dict[str, dict[str, Any]]:
    return {
        request["request_id"]: request
        for request in load_request_catalog()["requests"]
    }


def _disallowed_lookup() -> dict[str, dict[str, Any]]:
    return {
        behavior["behavior_id"]: behavior
        for behavior in load_agent_policy()["disallowed_behaviors"]
    }


def classify_governed_request(request_id_or_text: str) -> dict[str, Any]:
    normalized = request_id_or_text.strip().lower()
    request_lookup = _request_lookup()
    for request in request_lookup.values():
        if normalized in {
            request["request_id"].lower(),
            request["label"].lower(),
            request["controlled_request"].lower(),
        }:
            intent = _intent_lookup()[request["intent_id"]]
            return {
                "status": "supported",
                "request": request,
                "intent": intent,
                "classification_note": "Matched an explicit controlled request catalog entry.",
            }

    for behavior_id, terms in _DISALLOWED_TERMS.items():
        if any(term in normalized for term in terms):
            behavior = _disallowed_lookup().get(behavior_id)
            return {
                "status": "blocked",
                "request": None,
                "intent": None,
                "blocked_behavior": behavior,
                "classification_note": (
                    "Request resembles a disallowed behavior and will not be routed."
                ),
            }

    return {
        "status": "unsupported",
        "request": None,
        "intent": None,
        "classification_note": (
            "No controlled request catalog entry matched. Use a supported governed request."
        ),
    }


def build_agent_guardrails_summary() -> dict[str, Any]:
    policy = load_agent_policy()
    execution_policy = policy["execution_policy"]
    return {
        "execution_policy": execution_policy,
        "supported_intent_count": len(policy["supported_intents"]),
        "disallowed_behavior_count": len(policy["disallowed_behaviors"]),
        "guardrails": [
            "Preview/recommendation only; no workflow execution.",
            "Controlled request catalog only; no free-form agent prompt box.",
            "Routes must reference existing orchestration workflows and tasks.",
            "Secret handling is limited to environment-variable requirement labels.",
            "Human review checkpoints are surfaced before any future execution layer.",
        ],
    }


def build_agent_route_explanation(request_id: str) -> dict[str, Any]:
    classification = classify_governed_request(request_id)
    if classification["status"] != "supported":
        return classification
    request = classification["request"]
    intent = classification["intent"]
    workflow = get_workflow_definition(request["workflow_id"])
    return {
        "status": "supported",
        "request_id": request["request_id"],
        "intent_id": intent["intent_id"],
        "workflow_id": workflow["workflow_id"],
        "pace_phase": request["pace_phase"],
        "page_routes": request["page_routes"],
        "route_explanation": (
            f"Controlled request '{request['label']}' maps to intent "
            f"'{intent['intent_title']}' and workflow '{workflow['workflow_title']}'."
        ),
        "notes": request["notes"],
    }


def build_agent_required_inputs(request_id: str) -> dict[str, Any]:
    classification = classify_governed_request(request_id)
    if classification["status"] != "supported":
        return classification
    request = classification["request"]
    workflow = get_workflow_definition(request["workflow_id"])
    blockers = summarize_blockers(request["workflow_id"])
    task_inputs = []
    for task_id in request["task_ids"]:
        task = get_task_definition(task_id)
        task_inputs.append(
            {
                "task_id": task_id,
                "inputs": task["inputs"],
                "required_artifacts": task["required_artifacts"],
                "required_env_vars": task["required_env_vars"],
            }
        )
    return {
        "status": "supported",
        "request_inputs": request["required_inputs"],
        "workflow_required_artifacts": workflow["required_artifacts"],
        "workflow_required_env_vars": workflow["required_env_vars"],
        "task_inputs": task_inputs,
        "blockers": blockers,
    }


def build_agent_expected_outputs(request_id: str) -> dict[str, Any]:
    classification = classify_governed_request(request_id)
    if classification["status"] != "supported":
        return classification
    request = classification["request"]
    task_outputs = []
    for task_id in request["task_ids"]:
        task = get_task_definition(task_id)
        task_outputs.append({"task_id": task_id, "outputs": task["outputs"]})
    return {
        "status": "supported",
        "request_expected_outputs": request["expected_outputs"],
        "task_outputs": task_outputs,
    }


def build_agent_blocker_summary(request_id: str) -> dict[str, Any]:
    classification = classify_governed_request(request_id)
    if classification["status"] != "supported":
        return classification
    request = classification["request"]
    blockers = summarize_blockers(request["workflow_id"])
    return {
        "status": blockers["status"],
        "workflow_id": request["workflow_id"],
        "env_status": blockers["env_status"],
        "artifact_status": blockers["artifact_status"],
        "blocked_states": blockers["blocked_states"],
    }


def build_agent_plan_preview(request_id_or_text: str) -> dict[str, Any]:
    classification = classify_governed_request(request_id_or_text)
    if classification["status"] != "supported":
        return {
            **classification,
            "plan_steps": [],
            "guardrails": build_agent_guardrails_summary(),
            "execution_allowed": False,
        }

    request = classification["request"]
    intent = classification["intent"]
    workflow = get_workflow_definition(request["workflow_id"])
    runbook = build_runbook_view(request["workflow_id"])
    blockers = summarize_blockers(request["workflow_id"])
    selected_tasks = [get_task_definition(task_id) for task_id in request["task_ids"]]
    selected_task_ids = {task["task_id"] for task in selected_tasks}
    plan_steps = []
    for task in runbook["tasks"]:
        plan_steps.append(
            {
                "task_id": task["task_id"],
                "task_title": task["task_title"],
                "included_for_request": task["task_id"] in selected_task_ids,
                "task_kind": task["task_kind"],
                "runtime_mode": task["runtime_mode"],
                "mutates_repo_files": task["mutates_repo_files"],
                "human_review_required": task["human_review_required"],
                "dependencies": task["dependencies"],
                "expected_outputs": task["outputs"],
                "preview_note": (
                    "Mapped directly to selected controlled request."
                    if task["task_id"] in selected_task_ids
                    else "Workflow context task; not directly selected by this request."
                ),
            }
        )

    return {
        "status": "preview_ready",
        "classification": classification,
        "request": request,
        "intent": intent,
        "workflow": workflow,
        "pace_phase": request["pace_phase"],
        "workflow_id": request["workflow_id"],
        "task_ids": request["task_ids"],
        "page_routes": request["page_routes"],
        "governed_query": request["governed_query"],
        "plan_steps": plan_steps,
        "required_inputs": build_agent_required_inputs(request["request_id"]),
        "expected_outputs": build_agent_expected_outputs(request["request_id"]),
        "blockers": blockers,
        "review_checkpoints": request["review_checkpoints"],
        "guardrails": build_agent_guardrails_summary(),
        "execution_allowed": False,
        "execution_policy_note": (
            "The agent shell previews and recommends only. It does not execute workflows, "
            "trigger Airflow, mutate files, or run background jobs."
        ),
    }
