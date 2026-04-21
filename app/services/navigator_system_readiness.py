from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.navigator_agent_shell import build_agent_guardrails_summary
from app.services.navigator_loader import get_repo_root
from app.services.navigator_orchestration import list_workflows, summarize_blockers
from app.services.navigator_types import NavigatorRegistryValidationError

_APPROVAL_GATE_REQUIRED_FIELDS = {
    "gate_id",
    "gate_title",
    "applies_to",
    "status",
    "execution_allowed_in_streamlit",
    "human_review_required",
    "approval_required_before_execution",
    "blocked_when",
    "ready_when",
    "notes",
}
_READINESS_COMPONENT_REQUIRED_FIELDS = {
    "component_id",
    "component_title",
    "readiness_kind",
    "status_goal",
    "required_artifacts",
    "required_env_vars",
    "validation_scripts",
    "approval_gates",
    "ready_when",
    "blocked_when",
    "demo_message",
}
_OPENAI_KEY_ALIASES = ("RAG_STREAMLIT_OPENAI_API_KEY", "OPENAI_API_KEY")


def get_system_registry_paths() -> dict[str, Path]:
    root = get_repo_root() / "navigator" / "system"
    return {
        "approval_gates": root / "approval_gates.json",
        "system_readiness": root / "system_readiness.json",
    }


def clear_system_readiness_caches() -> None:
    load_approval_gates.cache_clear()
    load_system_readiness_registry.cache_clear()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise NavigatorRegistryValidationError(
            f"Required system registry '{label}' is missing at {path}."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise NavigatorRegistryValidationError(
            f"System registry '{label}' is not valid JSON: {exc}"
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


def _validate_approval_gates(registry: dict[str, Any]) -> None:
    gates = registry.get("approval_gates")
    if not isinstance(gates, list) or not gates:
        raise NavigatorRegistryValidationError(
            "approval_gates.json must contain a non-empty approval_gates list."
        )
    status_vocabulary = set(registry.get("status_vocabulary", []))
    required_statuses = {"ready", "review_needed", "blocked", "preview_only"}
    if not required_statuses.issubset(status_vocabulary):
        raise NavigatorRegistryValidationError(
            "approval_gates.json must define the shared status vocabulary."
        )
    _ensure_unique(gates, "gate_id", "approval gate")
    for gate in gates:
        missing = _APPROVAL_GATE_REQUIRED_FIELDS - set(gate)
        if missing:
            raise NavigatorRegistryValidationError(
                f"Approval gate {gate.get('gate_id', '<missing>')} is missing fields: {sorted(missing)}"
            )
        if gate["status"] not in status_vocabulary:
            raise NavigatorRegistryValidationError(
                f"Approval gate {gate['gate_id']} has unknown status {gate['status']!r}."
            )
        if gate["execution_allowed_in_streamlit"] is not False:
            raise NavigatorRegistryValidationError(
                f"Approval gate {gate['gate_id']} must not allow Streamlit execution."
            )
        for list_field in ("applies_to", "blocked_when", "ready_when"):
            if not isinstance(gate[list_field], list):
                raise NavigatorRegistryValidationError(
                    f"Approval gate {gate['gate_id']} field '{list_field}' must be a list."
                )


def _validate_readiness_registry(
    registry: dict[str, Any],
    approval_gates: dict[str, Any],
) -> None:
    components = registry.get("readiness_components")
    if not isinstance(components, list) or not components:
        raise NavigatorRegistryValidationError(
            "system_readiness.json must contain readiness_components."
        )
    gate_ids = {gate["gate_id"] for gate in approval_gates["approval_gates"]}
    _ensure_unique(components, "component_id", "readiness component")
    for component in components:
        missing = _READINESS_COMPONENT_REQUIRED_FIELDS - set(component)
        if missing:
            raise NavigatorRegistryValidationError(
                f"Readiness component {component.get('component_id', '<missing>')} is missing fields: {sorted(missing)}"
            )
        for list_field in (
            "required_artifacts",
            "required_env_vars",
            "validation_scripts",
            "approval_gates",
            "ready_when",
            "blocked_when",
        ):
            if not isinstance(component[list_field], list):
                raise NavigatorRegistryValidationError(
                    f"Readiness component {component['component_id']} field '{list_field}' must be a list."
                )
        for gate_id in component["approval_gates"]:
            if gate_id not in gate_ids:
                raise NavigatorRegistryValidationError(
                    f"Readiness component {component['component_id']} references unknown approval gate {gate_id!r}."
                )


@lru_cache(maxsize=1)
def load_approval_gates() -> dict[str, Any]:
    registry = _load_json(get_system_registry_paths()["approval_gates"], "approval_gates")
    _validate_approval_gates(registry)
    return registry


@lru_cache(maxsize=1)
def load_system_readiness_registry() -> dict[str, Any]:
    approval_gates = load_approval_gates()
    registry = _load_json(
        get_system_registry_paths()["system_readiness"],
        "system_readiness",
    )
    _validate_readiness_registry(registry, approval_gates)
    return registry


def load_system_readiness_registries() -> dict[str, Any]:
    return {
        "approval_gates": load_approval_gates(),
        "system_readiness": load_system_readiness_registry(),
    }


def _env_status(required_env_vars: list[str]) -> dict[str, Any]:
    if not required_env_vars:
        return {"status": "not_required", "missing": [], "label": "No API/env requirement"}
    if all(alias in required_env_vars for alias in _OPENAI_KEY_ALIASES):
        available = any(os.environ.get(alias) for alias in _OPENAI_KEY_ALIASES)
        return {
            "status": "ready" if available else "api_key_needed",
            "missing": [] if available else ["RAG_STREAMLIT_OPENAI_API_KEY or OPENAI_API_KEY"],
            "label": "API key present via environment alias"
            if available
            else "API key needed for live retrieval paths",
        }
    missing = [name for name in required_env_vars if not os.environ.get(name)]
    return {
        "status": "ready" if not missing else "blocked",
        "missing": missing,
        "label": "Required env vars present" if not missing else "Required env vars missing",
    }


def _artifact_status(paths: list[str]) -> dict[str, Any]:
    repo_root = get_repo_root()
    missing = [path for path in paths if not (repo_root / path).exists()]
    return {
        "status": "ready" if not missing else "blocked",
        "missing": missing,
    }


def _script_status(paths: list[str]) -> dict[str, Any]:
    repo_root = get_repo_root()
    missing = [path for path in paths if not (repo_root / path).exists()]
    return {
        "status": "ready" if not missing else "blocked",
        "missing": missing,
    }


def _component_status(
    component: dict[str, Any],
    artifact_status: dict[str, Any],
    env_status: dict[str, Any],
    script_status: dict[str, Any],
) -> str:
    if artifact_status["status"] == "blocked" or script_status["status"] == "blocked":
        return "blocked"
    if env_status["status"] == "api_key_needed":
        return "review_needed"
    if component["readiness_kind"] in {"preview_only_planner", "optional_local_scaffold"}:
        return "preview_only"
    return "ready"


def build_system_readiness_report() -> dict[str, Any]:
    registries = load_system_readiness_registries()
    gate_lookup = {
        gate["gate_id"]: gate
        for gate in registries["approval_gates"]["approval_gates"]
    }
    component_rows = []
    status_counts = {"ready": 0, "review_needed": 0, "blocked": 0, "preview_only": 0}
    for component in registries["system_readiness"]["readiness_components"]:
        artifacts = _artifact_status(component["required_artifacts"])
        env = _env_status(component["required_env_vars"])
        scripts = _script_status(component["validation_scripts"])
        status = _component_status(component, artifacts, env, scripts)
        status_counts[status] += 1
        component_rows.append(
            {
                "component_id": component["component_id"],
                "component_title": component["component_title"],
                "readiness_kind": component["readiness_kind"],
                "status": status,
                "status_label": status.replace("_", " ").title(),
                "required_artifacts": component["required_artifacts"],
                "missing_artifacts": artifacts["missing"],
                "required_env_vars": component["required_env_vars"],
                "missing_env_vars": env["missing"],
                "validation_scripts": component["validation_scripts"],
                "missing_validation_scripts": scripts["missing"],
                "approval_gates": [gate_lookup[gate_id] for gate_id in component["approval_gates"]],
                "human_review_required": any(
                    gate_lookup[gate_id]["human_review_required"]
                    for gate_id in component["approval_gates"]
                ),
                "execution_allowed_in_streamlit": False,
                "demo_message": component["demo_message"],
                "blocked_when": component["blocked_when"],
                "ready_when": component["ready_when"],
            }
        )

    demo_ready = status_counts["blocked"] == 0
    return {
        "status": "ready",
        "demo_ready": demo_ready,
        "status_counts": status_counts,
        "component_rows": component_rows,
        "approval_gates": registries["approval_gates"]["approval_gates"],
        "guardrails": build_agent_guardrails_summary()["guardrails"],
        "summary": {
            "component_count": len(component_rows),
            "approval_gate_count": len(registries["approval_gates"]["approval_gates"]),
            "demo_status": "Demo ready with governed caveats" if demo_ready else "Blocked",
            "shared_status_vocabulary": registries["approval_gates"]["status_vocabulary"],
            "governance_note": (
                "System readiness reports visibility and approval boundaries only. "
                "It does not execute workflows, trigger Airflow, or run background jobs."
            ),
        },
    }


def build_demo_readiness_checklist() -> dict[str, Any]:
    readiness = build_system_readiness_report()
    component_rows = readiness["component_rows"]
    items = [
        {
            "check_id": "registries_ready",
            "label": "Governed registries are present",
            "status": next(row["status"] for row in component_rows if row["component_id"] == "registry_foundation"),
        },
        {
            "check_id": "retrieval_pack_ready",
            "label": "Retrieval pack is present",
            "status": next(row["status"] for row in component_rows if row["component_id"] == "retrieval_pack"),
        },
        {
            "check_id": "reviewer_surfaces_ready",
            "label": "Reviewer surfaces are available with caveats surfaced",
            "status": next(row["status"] for row in component_rows if row["component_id"] == "reviewer_surfaces"),
        },
        {
            "check_id": "orchestration_contracts_ready",
            "label": "Orchestration contracts are inspectable",
            "status": next(row["status"] for row in component_rows if row["component_id"] == "orchestration_contracts"),
        },
        {
            "check_id": "agent_shell_preview_only",
            "label": "Agent shell is preview-only",
            "status": next(row["status"] for row in component_rows if row["component_id"] == "agent_shell"),
        },
    ]
    ready_or_preview = {"ready", "review_needed", "preview_only"}
    return {
        "status": "ready" if all(item["status"] in ready_or_preview for item in items) else "blocked",
        "items": items,
        "notes": [
            "Use Streamlit for governed inspection and demo narration only.",
            "Use CLI validation/build scripts for offline or API-backed tasks.",
            "Do not present fallback heuristics as final model probability.",
            "Do not treat the Airflow scaffold as production deployment.",
        ],
    }


def build_execution_eligibility_summary() -> dict[str, Any]:
    workflows = list_workflows()
    rows = []
    for workflow in workflows:
        blockers = summarize_blockers(workflow["workflow_id"])
        rows.append(
            {
                "workflow_id": workflow["workflow_id"],
                "workflow_title": workflow["workflow_title"],
                "runtime_mode": workflow["runtime_mode"],
                "scheduler_eligibility": workflow["scheduler_eligibility"],
                "mutates_repo_files": workflow["mutates_repo_files"],
                "human_review_required": workflow["human_review_required"],
                "streamlit_execution_allowed": False,
                "blocker_status": blockers["status"],
                "missing_artifacts": blockers["artifact_status"]["missing"],
                "missing_env_vars": blockers["env_status"]["missing"],
            }
        )
    return {
        "status": "ready",
        "workflow_rows": rows,
        "summary": {
            "workflow_count": len(rows),
            "streamlit_executable_workflows": 0,
            "mutating_workflows": sum(1 for row in rows if row["mutates_repo_files"]),
            "human_review_workflows": sum(1 for row in rows if row["human_review_required"]),
            "note": "No workflow is executable from Streamlit in WP16.",
        },
    }
