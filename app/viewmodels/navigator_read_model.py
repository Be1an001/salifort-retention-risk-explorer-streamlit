from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services import (
    NavigatorEmbeddingConfigurationError,
    NavigatorEmbeddingIndexError,
    NavigatorEmbeddingRequestError,
    NavigatorRetrievalIndexNotFoundError,
    OpenAIEmbeddingConfig,
    assemble_governed_answer,
    build_agent_plan_preview,
    build_demo_readiness_checklist,
    build_execution_eligibility_summary,
    build_runbook_view,
    build_system_readiness_report,
    get_controlled_agent_requests,
    get_disallowed_agent_behaviors,
    get_drift_items,
    get_execution_order,
    get_pace_phase,
    get_retrieval_evaluation_queries,
    get_runtime_governance_summary,
    get_repo_root,
    get_truth_entries,
    get_supported_agent_intents,
    list_tasks_for_workflow,
    list_workflows,
    load_all_navigator_registries,
    load_retrieval_pack,
    recommend_page_for_topic,
    retrieve_governed_chunks,
)

_PHASE_ORDER = ("plan", "analyze", "construct", "execute")
_DEFAULT_TOPIC_PREVIEWS = (
    "where the public model truth lives",
    "team exposure prioritization for managers",
    "threshold tradeoffs and confusion matrix",
    "why fallback is not the final model probability",
)
_TOPIC_CONFIG = {
    "public_model_choice": {
        "label": "Public model choice",
        "query_text": "where the public model truth lives",
        "summary": "Preserved public model framing for the deployed app layer.",
        "truth_domains": ["public_model_truth", "project_identity_truth"],
        "drift_ids": ["drift_public_selection_vs_rerun_leader"],
        "phase_name": "plan",
    },
    "threshold_logic": {
        "label": "Threshold logic",
        "query_text": "threshold tradeoffs and confusion matrix",
        "summary": "How threshold 0.29 is preserved and explained in the public operational flow.",
        "truth_domains": ["public_model_truth", "runtime_truth"],
        "drift_ids": ["drift_public_selection_vs_rerun_leader"],
        "phase_name": "construct",
    },
    "artifact_backed_runtime": {
        "label": "Artifact-backed runtime",
        "query_text": "artifact-backed runtime governance",
        "summary": "The runtime should read generated artifacts and stay offline-build oriented.",
        "truth_domains": ["runtime_truth", "orchestration_truth"],
        "drift_ids": [
            "drift_static_figures_vs_generated_tables",
            "drift_builder_reference_vs_local_reimplementation",
        ],
        "phase_name": "execute",
    },
    "fallback_vs_final_truth": {
        "label": "Fallback vs final model truth",
        "query_text": "why fallback is not the final model probability",
        "summary": "Why fallback screening must remain separate from the final weighted XGBoost probability.",
        "truth_domains": ["fallback_truth", "runtime_truth", "public_model_truth"],
        "drift_ids": ["drift_runtime_rows_vs_fallback_rows"],
        "phase_name": "execute",
    },
    "department_exposure": {
        "label": "Department exposure",
        "query_text": "team exposure prioritization for managers",
        "summary": "Department-level prioritization and review context for manager-facing workflow.",
        "truth_domains": ["runtime_truth"],
        "drift_ids": ["drift_static_figures_vs_generated_tables"],
        "phase_name": "execute",
    },
    "explainability_shap": {
        "label": "Explainability / SHAP",
        "query_text": "shap explanation and feature importance",
        "summary": "How explainability is represented as governed interpretation support.",
        "truth_domains": ["runtime_truth", "method_truth"],
        "drift_ids": [
            "drift_static_figures_vs_generated_tables",
            "drift_builder_reference_vs_local_reimplementation",
        ],
        "phase_name": "construct",
    },
    "workforce_explorer_usage": {
        "label": "Workforce Explorer usage",
        "query_text": "employee filtering and workforce explorer screening",
        "summary": "How the explorer should be used without confusing fallback heuristics with final model truth.",
        "truth_domains": ["runtime_truth", "fallback_truth"],
        "drift_ids": ["drift_runtime_rows_vs_fallback_rows"],
        "phase_name": "analyze",
    },
    "pace_workflow": {
        "label": "PACE workflow",
        "query_text": "pace workflow map",
        "summary": "How the project maps into Plan, Analyze, Construct, and Execute.",
        "truth_domains": ["method_truth", "project_identity_truth"],
        "drift_ids": ["drift_public_narrative_vs_snapshot_evidence"],
        "phase_name": "plan",
    },
    "known_drift_items": {
        "label": "Known drift items",
        "query_text": "known drift items and governance",
        "summary": "Governed drift that should be preserved and explained rather than flattened away.",
        "truth_domains": ["orchestration_truth", "runtime_truth"],
        "drift_ids": [
            "drift_public_selection_vs_rerun_leader",
            "drift_runtime_rows_vs_fallback_rows",
            "drift_builder_reference_vs_local_reimplementation",
            "drift_contract_optional_model_modes_schema",
            "drift_contract_employee_scores_extra_columns",
            "drift_public_narrative_vs_snapshot_evidence",
        ],
        "phase_name": "construct",
    },
}
_ROUTE_TITLES = {
    "overview": "Overview",
    "pace-navigator": "PACE Navigator",
    "workforce-explorer": "Workforce Explorer",
    "eda-patterns": "EDA & Patterns",
    "model-threshold-lab": "Model & Threshold Lab",
    "explainability": "Explainability",
    "manager-action-view": "Manager Action View",
    "methods-limitations": "Methods & Limitations",
}
_REVIEWER_SORT_OPTIONS = [
    {
        "key": "similarity_desc",
        "label": "Similarity score descending",
    },
    {
        "key": "authority_priority",
        "label": "Authority priority",
    },
    {
        "key": "truth_first",
        "label": "Truth-first grouping",
    },
    {
        "key": "drift_first",
        "label": "Drift-first grouping",
    },
    {
        "key": "retrieval_role",
        "label": "Retrieval role grouping",
    },
]
_PREVIEW_ALLOWED_EXTENSIONS = {
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_PREVIEW_BLOCKED_EXTENSIONS = {
    ".csv",
    ".jpeg",
    ".jpg",
    ".npy",
    ".parquet",
    ".png",
    ".webp",
}
_PREVIEW_MAX_BYTES = 80_000
_PREVIEW_CHAR_LIMIT = 12_000
_PREVIEW_SECRET_MARKERS = ("secret", "token", "credential", ".env")


def _first_truth_entry(domain: str) -> dict[str, Any]:
    matches = get_truth_entries(domain)
    if not matches:
        raise KeyError(f"No truth entry found for domain {domain!r}.")
    return matches[0]


def _all_truth_entries_lookup() -> dict[str, dict[str, Any]]:
    return {entry["domain"]: entry for entry in get_truth_entries()}


def _source_registry_lookup() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source_registry = load_all_navigator_registries().source_registry["sources"]
    by_path = {str(source["path"]): source for source in source_registry}
    return source_registry, by_path


def _drift_lookup() -> dict[str, dict[str, Any]]:
    return {item["drift_id"]: item for item in get_drift_items()}


@lru_cache(maxsize=1)
def _chunk_text_lookup() -> dict[str, str]:
    return {
        chunk["chunk_id"]: chunk["text"]
        for chunk in load_retrieval_pack()["chunks"]
    }


@lru_cache(maxsize=1)
def _chunk_record_lookup() -> dict[str, dict[str, Any]]:
    return {
        chunk["chunk_id"]: chunk
        for chunk in load_retrieval_pack()["chunks"]
    }


@lru_cache(maxsize=1)
def _retrieval_pack_manifest() -> dict[str, Any]:
    return load_retrieval_pack()["manifest"]


def _authority_priority(authority_level: str) -> int:
    normalized = authority_level.lower()
    if "canonical" in normalized or "public" in normalized:
        return 0
    if "primary" in normalized or "runtime" in normalized:
        return 1
    if "generated" in normalized or "artifact" in normalized:
        return 2
    if "reference" in normalized:
        return 4
    return 3


def _role_priority(retrieval_role: str) -> int:
    return 0 if retrieval_role == "answer_ready" else 1


def _has_any(values: list[str], selected: str | None) -> bool:
    return selected in (None, "All") or selected in values


def _as_markdown_list(items: list[str], empty_label: str = "None") -> str:
    if not items:
        return f"- {empty_label}"
    return "\n".join(f"- {item}" for item in items)


def _source_path_is_governed(path: str) -> bool:
    source_registry = load_all_navigator_registries().source_registry["sources"]
    source_paths = {str(source["path"]) for source in source_registry}
    chunk_source_paths = {
        source_path
        for chunk in load_retrieval_pack()["chunks"]
        for source_path in chunk["source_paths"]
    }
    return path in source_paths or path in chunk_source_paths


def _resolve_repo_local_path(path: str) -> Path | None:
    candidate = Path(path)
    if candidate.is_absolute():
        return None
    repo_root = get_repo_root().resolve()
    resolved = (repo_root / candidate).resolve()
    if repo_root != resolved and repo_root not in resolved.parents:
        return None
    return resolved


def evaluate_source_preview(path: str) -> dict[str, Any]:
    normalized_path = path.replace("\\", "/")
    suffix = Path(normalized_path).suffix.lower()
    lower_parts = [part.lower() for part in Path(normalized_path).parts]
    lower_path = normalized_path.lower()
    base_payload = {
        "source_path": normalized_path,
        "eligible": False,
        "status": "ineligible",
        "reason": "",
        "preview_text": "",
        "preview_char_count": 0,
        "file_size_bytes": None,
        "extension": suffix or "none",
        "is_truncated": False,
        "limit_note": (
            f"Preview is capped at {_PREVIEW_CHAR_LIMIT} characters and {_PREVIEW_MAX_BYTES} bytes."
        ),
    }

    if any(marker in lower_path for marker in _PREVIEW_SECRET_MARKERS):
        return {
            **base_payload,
            "reason": "Path is blocked because it looks secret-like or environment-related.",
        }
    if ".git" in lower_parts or "__pycache__" in lower_parts or ".venv" in lower_parts:
        return {
            **base_payload,
            "reason": "Path is blocked because hidden repo internals, caches, and local environments are never previewed.",
        }
    if suffix in _PREVIEW_BLOCKED_EXTENSIONS:
        return {
            **base_payload,
            "reason": f"Extension `{suffix}` is explicitly blocked for governed preview.",
        }
    resolved = _resolve_repo_local_path(normalized_path)
    if resolved is None:
        return {
            **base_payload,
            "reason": "Only relative repo-local governed paths are eligible for preview.",
        }
    if not _source_path_is_governed(normalized_path):
        return {
            **base_payload,
            "reason": "Source path is not present in the governed source registry or retrieval pack.",
        }
    if suffix not in _PREVIEW_ALLOWED_EXTENSIONS:
        return {
            **base_payload,
            "reason": f"Extension `{suffix or 'none'}` is not in the allowed text-like preview list.",
        }

    if not resolved.exists() or not resolved.is_file():
        return {
            **base_payload,
            "reason": "Local file does not exist as a readable repo file.",
        }

    file_size = resolved.stat().st_size
    payload = {**base_payload, "file_size_bytes": file_size}
    if file_size > _PREVIEW_MAX_BYTES:
        return {
            **payload,
            "reason": f"File is larger than the governed preview size limit of {_PREVIEW_MAX_BYTES} bytes.",
        }

    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            **payload,
            "reason": "File is not valid UTF-8 text, so it is not previewed.",
        }

    truncated = len(text) > _PREVIEW_CHAR_LIMIT
    preview_text = text[:_PREVIEW_CHAR_LIMIT]
    return {
        **payload,
        "eligible": True,
        "status": "eligible",
        "reason": "Governed repo-local text-like file is eligible for read-only preview.",
        "preview_text": preview_text,
        "preview_char_count": len(preview_text),
        "is_truncated": truncated,
    }


def _size_bucket(file_size_bytes: int | None) -> str:
    if file_size_bytes is None:
        return "unknown"
    if file_size_bytes <= 10_000:
        return "small"
    if file_size_bytes <= _PREVIEW_MAX_BYTES:
        return "preview_limit"
    return "large_or_blocked"


def build_eligible_source_index() -> dict[str, Any]:
    source_registry = load_all_navigator_registries().source_registry["sources"]
    retrieval_chunks = load_retrieval_pack()["chunks"]
    metadata_by_path: dict[str, dict[str, Any]] = {}

    def ensure_path(path: str) -> dict[str, Any]:
        normalized = str(path).replace("\\", "/")
        if normalized not in metadata_by_path:
            metadata_by_path[normalized] = {
                "source_path": normalized,
                "source_ids": [],
                "source_titles": [],
                "source_kinds": [],
                "repo_layers": [],
                "authority_levels": [],
                "consumer_pages": [],
                "document_ids": [],
                "chunk_ids": [],
                "source_universe": set(),
            }
        return metadata_by_path[normalized]

    for source in source_registry:
        record = ensure_path(source["path"])
        record["source_universe"].add("source_registry")
        record["source_ids"].append(source["source_id"])
        record["source_titles"].append(source["title"])
        record["source_kinds"].append(source["source_kind"])
        record["repo_layers"].append(source["repo_layer"])
        record["authority_levels"].append(source["authority_level"])
        record["consumer_pages"].extend(source.get("consumer_pages", []))

    for chunk in retrieval_chunks:
        for source_path in chunk["source_paths"]:
            record = ensure_path(source_path)
            record["source_universe"].add("retrieval_pack")
            record["document_ids"].append(chunk["document_id"])
            record["chunk_ids"].append(chunk["chunk_id"])
            if chunk.get("authority_level"):
                record["authority_levels"].append(chunk["authority_level"])

    rows: list[dict[str, Any]] = []
    for source_path, metadata in metadata_by_path.items():
        preview = evaluate_source_preview(source_path)
        extension = preview["extension"]
        source_universe = sorted(metadata["source_universe"])
        source_titles = sorted(set(metadata["source_titles"]))
        source_kinds = sorted(set(metadata["source_kinds"]))
        authority_levels = sorted(set(metadata["authority_levels"]))
        repo_layers = sorted(set(metadata["repo_layers"]))
        consumer_pages = sorted(set(metadata["consumer_pages"]))
        document_ids = sorted(set(metadata["document_ids"]))
        chunk_ids = sorted(set(metadata["chunk_ids"]))
        rows.append(
            {
                "source_path": source_path,
                "source_label": source_titles[0] if source_titles else Path(source_path).name,
                "source_type": ", ".join(source_kinds) if source_kinds else "retrieval_pack_source",
                "source_universe": ", ".join(source_universe),
                "preview_eligible": bool(preview["eligible"]),
                "preview_status": preview["status"],
                "blocked_reason": "" if preview["eligible"] else preview["reason"],
                "preview_reason": preview["reason"],
                "extension": extension,
                "file_size_bytes": preview["file_size_bytes"],
                "size_bucket": _size_bucket(preview["file_size_bytes"]),
                "preview_class": "previewable_text" if preview["eligible"] else "blocked_or_unavailable",
                "source_ids": sorted(set(metadata["source_ids"])),
                "document_ids": document_ids,
                "chunk_count": len(chunk_ids),
                "authority_levels": authority_levels,
                "repo_layers": repo_layers,
                "consumer_pages": consumer_pages,
            }
        )

    rows.sort(
        key=lambda item: (
            0 if item["preview_eligible"] else 1,
            str(item["source_path"]).lower(),
        )
    )
    eligible_rows = [row for row in rows if row["preview_eligible"]]
    blocked_rows = [row for row in rows if not row["preview_eligible"]]
    return {
        "status": "ready",
        "rows": rows,
        "summary": {
            "total_sources": len(rows),
            "eligible_sources": len(eligible_rows),
            "blocked_sources": len(blocked_rows),
            "source_registry_count": len(source_registry),
            "retrieval_pack_chunk_count": len(retrieval_chunks),
            "allowed_extensions": sorted(_PREVIEW_ALLOWED_EXTENSIONS),
            "blocked_extensions": sorted(_PREVIEW_BLOCKED_EXTENSIONS),
            "preview_max_bytes": _PREVIEW_MAX_BYTES,
            "preview_char_limit": _PREVIEW_CHAR_LIMIT,
            "governance_note": (
                "This index is limited to governed source-registry and retrieval-pack paths. "
                "It is not arbitrary repository enumeration or unrestricted file browsing."
            ),
        },
    }


def build_orchestration_summary() -> dict[str, Any]:
    workflows = list_workflows()
    workflow_cards: list[dict[str, Any]] = []
    runtime_modes: set[str] = set()
    scheduler_classes: set[str] = set()
    task_count = 0
    mutating_workflows = 0
    human_review_workflows = 0

    for workflow in workflows:
        tasks = list_tasks_for_workflow(workflow["workflow_id"])
        runbook = build_runbook_view(workflow["workflow_id"])
        blocker_status = runbook["blockers"]["status"]
        runtime_modes.add(workflow["runtime_mode"])
        scheduler_classes.add(workflow["scheduler_eligibility"])
        task_count += len(tasks)
        if workflow["mutates_repo_files"]:
            mutating_workflows += 1
        if workflow["human_review_required"]:
            human_review_workflows += 1
        workflow_cards.append(
            {
                "workflow_id": workflow["workflow_id"],
                "workflow_title": workflow["workflow_title"],
                "workflow_kind": workflow["workflow_kind"],
                "stage": workflow["stage"],
                "phase": workflow["phase"],
                "runtime_mode": workflow["runtime_mode"],
                "scheduler_eligibility": workflow["scheduler_eligibility"],
                "human_review_required": workflow["human_review_required"],
                "mutates_repo_files": workflow["mutates_repo_files"],
                "task_count": len(tasks),
                "execution_order": get_execution_order(workflow["workflow_id"]),
                "blocker_status": blocker_status,
                "missing_artifacts": runbook["blockers"]["artifact_status"]["missing"],
                "missing_env_vars": runbook["blockers"]["env_status"]["missing"],
                "notes": workflow["notes"],
            }
        )

    return {
        "status": "ready",
        "summary": {
            "workflow_count": len(workflows),
            "workflow_task_memberships": task_count,
            "mutating_workflows": mutating_workflows,
            "human_review_workflows": human_review_workflows,
            "runtime_modes": sorted(runtime_modes),
            "scheduler_classes": sorted(scheduler_classes),
            "governance_note": (
                "The orchestration layer defines task/workflow contracts only. "
                "It does not execute jobs from Streamlit and does not add agent behavior."
            ),
        },
        "workflow_cards": workflow_cards,
    }


def build_orchestration_workflow_detail(workflow_id: str) -> dict[str, Any]:
    runbook = build_runbook_view(workflow_id)
    task_rows = []
    for index, task in enumerate(runbook["tasks"], start=1):
        task_rows.append(
            {
                "order": index,
                "task_id": task["task_id"],
                "task_title": task["task_title"],
                "task_kind": task["task_kind"],
                "phase": task["phase"],
                "runtime_mode": task["runtime_mode"],
                "scheduler_eligibility": task["scheduler_eligibility"],
                "mutates_repo_files": task["mutates_repo_files"],
                "human_review_required": task["human_review_required"],
                "dependencies": task["dependencies"],
                "required_artifacts": task["required_artifacts"],
                "required_env_vars": task["required_env_vars"],
                "blocked_states": task["blocked_states"],
                "command_hint": task.get("command_hint", ""),
                "notes": task["notes"],
            }
        )
    return {
        "status": "ready",
        "workflow": runbook["workflow"],
        "summary": runbook["summary"],
        "blockers": runbook["blockers"],
        "task_rows": task_rows,
    }


def build_agent_shell_context() -> dict[str, Any]:
    requests = get_controlled_agent_requests()
    intents = get_supported_agent_intents()
    disallowed = get_disallowed_agent_behaviors()
    return {
        "status": "ready",
        "requests": requests,
        "intents": intents,
        "disallowed_behaviors": disallowed,
        "summary": {
            "controlled_request_count": len(requests),
            "supported_intent_count": len(intents),
            "disallowed_behavior_count": len(disallowed),
            "governance_note": (
                "The governed PACE agent shell classifies controlled requests and "
                "builds preview plans only. It does not execute workflows or trigger Airflow."
            ),
        },
    }


def build_agent_shell_preview(request_id: str) -> dict[str, Any]:
    preview = build_agent_plan_preview(request_id)
    if preview["status"] != "preview_ready":
        return preview

    plan_rows = [
        {
            "order": index,
            "task_id": step["task_id"],
            "task_title": step["task_title"],
            "included_for_request": step["included_for_request"],
            "task_kind": step["task_kind"],
            "runtime_mode": step["runtime_mode"],
            "mutates_repo_files": step["mutates_repo_files"],
            "human_review_required": step["human_review_required"],
            "dependencies": step["dependencies"],
            "preview_note": step["preview_note"],
        }
        for index, step in enumerate(preview["plan_steps"], start=1)
    ]
    route_summary = {
        "request_id": preview["request"]["request_id"],
        "label": preview["request"]["label"],
        "intent_id": preview["intent"]["intent_id"],
        "intent_title": preview["intent"]["intent_title"],
        "workflow_id": preview["workflow_id"],
        "workflow_title": preview["workflow"]["workflow_title"],
        "pace_phase": preview["pace_phase"],
        "page_routes": preview["page_routes"],
        "governed_query": preview["governed_query"],
        "execution_allowed": preview["execution_allowed"],
        "blocker_status": preview["blockers"]["status"],
        "missing_artifacts": preview["blockers"]["artifact_status"]["missing"],
        "missing_env_vars": preview["blockers"]["env_status"]["missing"],
    }
    return {
        **preview,
        "route_summary": route_summary,
        "plan_rows": plan_rows,
    }


def build_final_system_readiness_context() -> dict[str, Any]:
    readiness = build_system_readiness_report()
    checklist = build_demo_readiness_checklist()
    execution = build_execution_eligibility_summary()
    component_cards = [
        {
            "component_id": row["component_id"],
            "component_title": row["component_title"],
            "readiness_kind": row["readiness_kind"],
            "status": row["status"],
            "status_label": row["status_label"],
            "human_review_required": row["human_review_required"],
            "execution_allowed_in_streamlit": row["execution_allowed_in_streamlit"],
            "missing_artifacts": row["missing_artifacts"],
            "missing_env_vars": row["missing_env_vars"],
            "demo_message": row["demo_message"],
            "approval_gate_titles": [
                gate["gate_title"] for gate in row["approval_gates"]
            ],
        }
        for row in readiness["component_rows"]
    ]
    return {
        "status": "ready",
        "summary": readiness["summary"],
        "demo_ready": readiness["demo_ready"],
        "status_counts": readiness["status_counts"],
        "component_cards": component_cards,
        "approval_gates": readiness["approval_gates"],
        "demo_checklist": checklist,
        "execution_eligibility": execution,
        "guardrails": readiness["guardrails"],
    }


def get_project_identity_card() -> dict[str, Any]:
    identity_truth = _first_truth_entry("project_identity_truth")
    method_truth = _first_truth_entry("method_truth")
    return {
        "title": "What This Navigator Is",
        "summary": identity_truth["description"],
        "highlights": [
            "This repo is the newer public Streamlit app layer, not the legacy notebook-first modeling repo.",
            "The navigator is a governed explainer for truth, drift, and phase mapping.",
            "PACE is the future navigation spine for understanding how the project is organized.",
        ],
        "supporting_truth_ids": [
            identity_truth["truth_id"],
            method_truth["truth_id"],
        ],
    }


def get_public_model_truth_card() -> dict[str, Any]:
    public_truth = _first_truth_entry("public_model_truth")
    return {
        "title": "Current Public Truth",
        "model_name": "Weighted XGBoost",
        "selected_threshold": "0.29",
        "summary": public_truth["description"],
        "authority_rule": public_truth["authority_rule"],
        "preserve_in_upgrade": bool(public_truth["preserve_in_upgrade"]),
        "supporting_truth_id": public_truth["truth_id"],
    }


def get_runtime_governance_cards() -> list[dict[str, Any]]:
    governance_summary = get_runtime_governance_summary()
    return [
        {
            "title": "Artifact-Backed Runtime",
            "summary": governance_summary["artifact_backed_runtime_truth"]["description"],
            "authority_rule": governance_summary["artifact_backed_runtime_truth"][
                "authority_rule"
            ],
            "tone": "primary",
        },
        {
            "title": "Fallback Separation",
            "summary": governance_summary["fallback_truth"]["description"],
            "authority_rule": governance_summary["fallback_truth"]["authority_rule"],
            "tone": "warning",
        },
    ]


def get_drift_highlight_cards() -> list[dict[str, Any]]:
    tracked_ids = [
        "drift_public_selection_vs_rerun_leader",
        "drift_runtime_rows_vs_fallback_rows",
        "drift_builder_reference_vs_local_reimplementation",
    ]
    drift_lookup = {item["drift_id"]: item for item in get_drift_items()}
    cards = []
    for drift_id in tracked_ids:
        item = drift_lookup.get(drift_id)
        if item is None:
            continue
        cards.append(
            {
                "title": item["title"],
                "severity": str(item["severity"]).capitalize(),
                "status": str(item["status"]).replace("_", " ").capitalize(),
                "user_visible_risk": item["user_visible_risk"],
                "upgrade_handling_rule": item["upgrade_handling_rule"],
            }
        )
    return cards


def get_pace_phase_cards() -> list[dict[str, Any]]:
    cards = []
    for phase_name in _PHASE_ORDER:
        phase = get_pace_phase(phase_name)
        cards.append(
            {
                "phase_id": phase["phase_id"],
                "phase_title": phase["phase_title"],
                "phase_goal": phase["phase_goal"],
                "portfolio_meaning": phase["portfolio_meaning"],
                "app_pages": phase["app_pages"],
                "known_drifts": phase["known_drifts"],
                "future_navigator_use": phase["future_navigator_use"],
            }
        )
    return cards


def get_example_topic_recommendations() -> list[dict[str, Any]]:
    previews = []
    for topic in _DEFAULT_TOPIC_PREVIEWS:
        recommendation = recommend_page_for_topic(topic)
        previews.append(
            {
                "topic": topic,
                "recommended_page_title": recommendation["recommended_page_title"],
                "recommended_page_route": recommendation["recommended_page_route"],
                "supporting_phase": recommendation["supporting_phase"],
                "reason": recommendation["reason"],
            }
        )
    return previews


def get_topic_options() -> list[dict[str, str]]:
    return [
        {
            "topic_key": topic_key,
            "label": topic_config["label"],
            "summary": topic_config["summary"],
        }
        for topic_key, topic_config in _TOPIC_CONFIG.items()
    ]


def _build_truth_summaries(truth_domains: list[str]) -> list[dict[str, Any]]:
    truth_lookup = _all_truth_entries_lookup()
    summaries = []
    for domain in truth_domains:
        entry = truth_lookup.get(domain)
        if entry is None:
            continue
        summaries.append(
            {
                "domain": domain,
                "truth_id": entry["truth_id"],
                "title": entry["title"],
                "description": entry["description"],
                "authority_rule": entry["authority_rule"],
                "primary_sources": entry["primary_sources"],
                "secondary_sources": entry["secondary_sources"],
            }
        )
    return summaries


def _build_source_drilldown(truth_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_registry, source_by_path = _source_registry_lookup()
    seen_source_ids: set[str] = set()
    source_records: list[dict[str, Any]] = []

    def add_record(path: str, role: str) -> None:
        source = source_by_path.get(path)
        if source is None:
            source_records.append(
                {
                    "source_id": f"external::{path}",
                    "title": path,
                    "role": role,
                    "source_kind": "external_reference",
                    "path": path,
                    "repo_layer": "external",
                    "authority_level": "external_reference",
                    "canonical_scope": [],
                    "runtime_scope": [],
                    "consumer_pages": [],
                    "notes": "Referenced by truth registry but not present in the in-repo source registry.",
                }
            )
            return

        if source["source_id"] in seen_source_ids:
            return
        seen_source_ids.add(source["source_id"])
        source_records.append(
            {
                "source_id": source["source_id"],
                "title": source["title"],
                "role": role,
                "source_kind": source["source_kind"],
                "path": source["path"],
                "repo_layer": source["repo_layer"],
                "authority_level": source["authority_level"],
                "canonical_scope": source["canonical_scope"],
                "runtime_scope": source["runtime_scope"],
                "consumer_pages": source["consumer_pages"],
                "notes": source["notes"],
            }
        )

    for truth in truth_summaries:
        for path in truth["primary_sources"]:
            add_record(path, "primary")
        for path in truth["secondary_sources"]:
            add_record(path, "secondary")

    source_records.sort(
        key=lambda item: (
            0 if item["role"] == "primary" else 1,
            str(item["title"]).lower(),
        )
    )
    return source_records


def _build_topic_drift_records(drift_ids: list[str]) -> list[dict[str, Any]]:
    lookup = _drift_lookup()
    records = []
    for drift_id in drift_ids:
        item = lookup.get(drift_id)
        if item is None:
            continue
        records.append(
            {
                "drift_id": item["drift_id"],
                "title": item["title"],
                "severity": str(item["severity"]).capitalize(),
                "status": str(item["status"]).replace("_", " ").capitalize(),
                "canonical_side": item["canonical_side"],
                "current_side": item["current_side"],
                "source_evidence": item["source_evidence"],
                "user_visible_risk": item["user_visible_risk"],
                "upgrade_handling_rule": item["upgrade_handling_rule"],
                "notes": item["notes"],
            }
        )
    return records


def build_navigator_topic_drilldown(topic_key: str) -> dict[str, Any]:
    topic_config = _TOPIC_CONFIG.get(topic_key)
    if topic_config is None:
        raise KeyError(f"Unknown navigator topic key {topic_key!r}.")

    recommendation = recommend_page_for_topic(topic_config["query_text"])
    truth_summaries = _build_truth_summaries(topic_config["truth_domains"])
    source_records = _build_source_drilldown(truth_summaries)
    drift_records = _build_topic_drift_records(topic_config["drift_ids"])
    phase = get_pace_phase(topic_config["phase_name"])

    return {
        "topic_key": topic_key,
        "topic_label": topic_config["label"],
        "topic_summary": topic_config["summary"],
        "supporting_phase": {
            "phase_id": phase["phase_id"],
            "phase_title": phase["phase_title"],
            "phase_goal": phase["phase_goal"],
        },
        "routing_recommendation": recommendation,
        "truth_summaries": truth_summaries,
        "source_records": source_records,
        "drift_records": drift_records,
    }


def get_drift_records(
    severity: str | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    records = []
    for item in get_drift_items(severity=severity, status=status):
        records.append(
            {
                "drift_id": item["drift_id"],
                "title": item["title"],
                "severity": str(item["severity"]).capitalize(),
                "status": str(item["status"]).replace("_", " ").capitalize(),
                "canonical_side": item["canonical_side"],
                "current_side": item["current_side"],
                "source_evidence": item["source_evidence"],
                "user_visible_risk": item["user_visible_risk"],
                "upgrade_handling_rule": item["upgrade_handling_rule"],
                "notes": item["notes"],
            }
        )
    return records


def get_available_drift_filters() -> dict[str, list[str]]:
    drift_items = get_drift_items()
    return {
        "severities": sorted(
            {str(item["severity"]).capitalize() for item in drift_items}
        ),
        "statuses": sorted(
            {str(item["status"]).replace("_", " ").capitalize() for item in drift_items}
        ),
    }


def build_navigator_page_context() -> dict[str, Any]:
    method_truth = _first_truth_entry("method_truth")
    return {
        "page_title": "PACE Navigator",
        "page_caption": (
            "Read-only project navigator for governed truth, drift, runtime framing, and "
            "phase-aware orientation."
        ),
        "project_identity_card": get_project_identity_card(),
        "public_model_truth_card": get_public_model_truth_card(),
        "runtime_governance_cards": get_runtime_governance_cards(),
        "drift_highlight_cards": get_drift_highlight_cards(),
        "pace_phase_cards": get_pace_phase_cards(),
        "topic_recommendations": get_example_topic_recommendations(),
        "topic_options": get_topic_options(),
        "default_topic_key": "artifact_backed_runtime",
        "drift_filters": get_available_drift_filters(),
        "pace_spine_note": method_truth["description"],
    }


def get_governed_answer_query_options() -> list[dict[str, str]]:
    return [
        {
            "query": spec.query,
            "description": spec.description,
        }
        for spec in get_retrieval_evaluation_queries()
    ]


def build_governed_answer_view(query: str, *, top_k: int = 8) -> dict[str, Any]:
    try:
        config = OpenAIEmbeddingConfig.from_env()
    except NavigatorEmbeddingConfigurationError as exc:
        return {
            "status": "blocked",
            "error_kind": "missing_api_key",
            "message": str(exc),
            "query": query,
        }

    try:
        retrieval_results = retrieve_governed_chunks(query, config=config, top_k=top_k)
        answer = assemble_governed_answer(
            query,
            config=config,
            top_k=top_k,
            retrieved_results=retrieval_results,
        ).as_dict()
    except NavigatorRetrievalIndexNotFoundError as exc:
        return {
            "status": "blocked",
            "error_kind": "missing_index",
            "message": str(exc),
            "query": query,
        }
    except NavigatorEmbeddingRequestError as exc:
        return {
            "status": "error",
            "error_kind": "embedding_request_failed",
            "message": str(exc),
            "query": query,
        }
    except NavigatorEmbeddingIndexError as exc:
        return {
            "status": "error",
            "error_kind": "retrieval_runtime_error",
            "message": str(exc),
            "query": query,
        }

    retrieval_rows = []
    chunk_texts = _chunk_text_lookup()
    for index, item in enumerate(retrieval_results, start=1):
        retrieval_rows.append(
            {
                "rank": index,
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "title": item["title"],
                "similarity_score": round(float(item["similarity_score"]), 4),
                "retrieval_role": item["retrieval_role"],
                "truth_tags": item["truth_tags"],
                "drift_tags": item["drift_tags"],
                "phase_tags": item["phase_tags"],
                "page_routes": item["page_routes"],
                "page_titles": [_ROUTE_TITLES.get(route, route) for route in item["page_routes"]],
                "source_paths": item["source_paths"],
                "registry_refs": item["registry_refs"],
                "authority_level": item["authority_level"],
                "text_preview": item["text_preview"],
                "full_text": chunk_texts.get(item["chunk_id"], item["text_preview"]),
                "caveats": item["caveats"],
            }
        )

    citation_rows = []
    retrieval_by_chunk = {item["chunk_id"]: item for item in retrieval_rows}
    for citation in answer["citations"]:
        retrieval_match = retrieval_by_chunk.get(citation["chunk_id"], {})
        citation_rows.append(
            {
                "chunk_id": citation["chunk_id"],
                "document_id": citation["document_id"],
                "title": citation["title"],
                "similarity_score": retrieval_match.get("similarity_score"),
                "source_paths": citation["source_paths"],
                "registry_refs": citation["registry_refs"],
                "truth_tags": citation["truth_tags"],
                "drift_tags": citation["drift_tags"],
                "phase_tags": citation["phase_tags"],
                "page_routes": retrieval_match.get("page_routes", []),
                "page_titles": retrieval_match.get("page_titles", []),
                "retrieval_role": citation["retrieval_role"],
                "authority_level": retrieval_match.get("authority_level", "unknown"),
                "text_preview": retrieval_match.get("text_preview", ""),
                "full_text": retrieval_match.get("full_text", ""),
                "caveats": retrieval_match.get("caveats", []),
            }
        )

    return {
        "status": "ready",
        "query": query,
        "answer": answer,
        "retrieval_rows": retrieval_rows,
        "citation_rows": citation_rows,
        "query_options": get_governed_answer_query_options(),
    }


def get_reviewer_sort_options() -> list[dict[str, str]]:
    return list(_REVIEWER_SORT_OPTIONS)


def build_reviewer_filter_options(
    retrieval_rows: list[dict[str, Any]]
) -> dict[str, list[str]]:
    def collect(key: str) -> list[str]:
        values: set[str] = set()
        for item in retrieval_rows:
            raw_value = item[key]
            if isinstance(raw_value, list):
                values.update(str(value) for value in raw_value)
            elif raw_value:
                values.add(str(raw_value))
        return ["All"] + sorted(values)

    return {
        "truth_tags": collect("truth_tags"),
        "drift_tags": collect("drift_tags"),
        "phase_tags": collect("phase_tags"),
        "retrieval_roles": collect("retrieval_role"),
        "page_routes": collect("page_routes"),
        "authority_levels": collect("authority_level"),
    }


def filter_and_sort_retrieval_rows(
    retrieval_rows: list[dict[str, Any]],
    *,
    truth_tag: str | None = None,
    drift_tag: str | None = None,
    phase_tag: str | None = None,
    retrieval_role: str | None = None,
    page_route: str | None = None,
    authority_level: str | None = None,
    sort_key: str = "similarity_desc",
) -> list[dict[str, Any]]:
    filtered = [
        item
        for item in retrieval_rows
        if _has_any(item["truth_tags"], truth_tag)
        and _has_any(item["drift_tags"], drift_tag)
        and _has_any(item["phase_tags"], phase_tag)
        and _has_any(item["page_routes"], page_route)
        and retrieval_role in (None, "All", item["retrieval_role"])
        and authority_level in (None, "All", item["authority_level"])
    ]

    sorters = {
        "similarity_desc": lambda item: (-float(item["similarity_score"]), item["rank"]),
        "authority_priority": lambda item: (
            _authority_priority(str(item["authority_level"])),
            -float(item["similarity_score"]),
            item["rank"],
        ),
        "truth_first": lambda item: (
            0 if item["truth_tags"] else 1,
            -float(item["similarity_score"]),
            item["rank"],
        ),
        "drift_first": lambda item: (
            0 if item["drift_tags"] else 1,
            -float(item["similarity_score"]),
            item["rank"],
        ),
        "retrieval_role": lambda item: (
            _role_priority(str(item["retrieval_role"])),
            -float(item["similarity_score"]),
            item["rank"],
        ),
    }
    sorter = sorters.get(sort_key, sorters["similarity_desc"])
    return sorted(filtered, key=sorter)


def build_citation_comparison(
    citation_rows: list[dict[str, Any]],
    left_chunk_id: str,
    right_chunk_id: str,
) -> dict[str, Any]:
    citation_lookup = {item["chunk_id"]: item for item in citation_rows}
    left = citation_lookup.get(left_chunk_id)
    right = citation_lookup.get(right_chunk_id)
    return {
        "status": "ready" if left and right else "incomplete",
        "left": left,
        "right": right,
    }


def build_support_quality_review(answer_view: dict[str, Any]) -> dict[str, Any]:
    if answer_view["status"] != "ready":
        return {
            "status_label": "Blocked",
            "tone": "warning",
            "indicators": [],
            "review_notes": [answer_view.get("message", "Answer view is not ready.")],
        }

    answer = answer_view["answer"]
    retrieval_rows = answer_view["retrieval_rows"]
    coverage = answer["coverage_summary"]
    query = str(answer_view["query"]).lower()
    canonical_truth_present = any(item["truth_tags"] for item in retrieval_rows)
    drift_context_present = any(item["drift_tags"] for item in retrieval_rows)
    page_route_present = any(item["page_routes"] for item in retrieval_rows)
    reference_only_count = sum(
        1 for item in retrieval_rows if item["retrieval_role"] == "reference_only"
    )

    if coverage["status"] == "sufficient" and canonical_truth_present:
        status_label = "Strong governed support"
        tone = "primary"
    elif coverage["status"] == "sufficient":
        status_label = "Sufficient routed support"
        tone = "neutral"
    elif canonical_truth_present or drift_context_present:
        status_label = "Partial governed support"
        tone = "warning"
    else:
        status_label = "Needs reviewer attention"
        tone = "warning"

    indicators = [
        {
            "label": "Canonical truth present",
            "value": "Yes" if canonical_truth_present else "No",
        },
        {
            "label": "Drift context present",
            "value": "Yes" if drift_context_present else "No",
        },
        {
            "label": "Page-route support present",
            "value": "Yes" if page_route_present else "No",
        },
        {
            "label": "Reference-only chunks",
            "value": str(reference_only_count),
        },
    ]

    review_notes: list[str] = []
    if "threshold 0.29" in query:
        review_notes.append(
            "Threshold rationale is stitched from governed public-truth and threshold-artifact support; review citations before treating it as a single-source rationale."
        )
    if "fallback" in query and not drift_context_present:
        review_notes.append(
            "Fallback comparison should include drift context; inspect retrieval filters if it is absent."
        )
    if coverage["status"] != "sufficient":
        review_notes.append(
            "Coverage is not marked sufficient; use the citations and retrieval inspector before relying on this answer."
        )
    if reference_only_count:
        review_notes.append(
            "Reference-only chunks are present; they can support review but should not override canonical truth."
        )
    if not review_notes:
        review_notes.append(
            "Support signals are complete for the fixed governed scenario; still inspect citations for provenance."
        )

    return {
        "status_label": status_label,
        "tone": tone,
        "indicators": indicators,
        "review_notes": review_notes,
    }


def _checklist_item(
    item_id: str,
    label: str,
    status: str,
    detail: str,
) -> dict[str, str]:
    return {
        "item_id": item_id,
        "label": label,
        "status": status,
        "detail": detail,
    }


def build_audit_checklist(
    answer_view: dict[str, Any],
    support_review: dict[str, Any],
    source_detail: dict[str, Any] | None = None,
    export_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixed_queries = {item["query"] for item in get_governed_answer_query_options()}
    query = str(answer_view.get("query", ""))
    answer_ready = answer_view.get("status") == "ready"
    source_detail_ready = bool(source_detail and source_detail.get("status") == "ready")
    source_preview_checked = bool(
        source_detail_ready and source_detail.get("preview_options") is not None
    )
    has_preview_signal = bool(
        source_detail_ready and source_detail.get("preview_status") in {"eligible", "ineligible"}
    )
    export_ready = bool(export_payload and export_payload.get("status") == "ready")

    if answer_ready:
        answer = answer_view["answer"]
        retrieval_rows = answer_view["retrieval_rows"]
        citation_rows = answer_view["citation_rows"]
        direct_answer = str(answer.get("direct_answer", "")).strip()
        truth_present = any(row["truth_tags"] for row in retrieval_rows)
        drift_present = any(row["drift_tags"] for row in retrieval_rows) or bool(
            answer.get("drift_and_caveats")
        )
        coverage_status = answer["coverage_summary"]["status"]
    else:
        direct_answer = ""
        citation_rows = []
        truth_present = False
        drift_present = False
        coverage_status = "blocked"

    items = [
        _checklist_item(
            "fixed_governed_query",
            "Query is from the fixed governed set",
            "ready" if query in fixed_queries else "blocked",
            (
                "Selected query is controlled by the governed scenario set."
                if query in fixed_queries
                else "Query is not in the fixed governed scenario set."
            ),
        ),
        _checklist_item(
            "direct_answer_present",
            "Direct governed answer is present",
            "ready" if direct_answer else "blocked",
            (
                "Deterministic answer assembly produced a direct governed answer."
                if direct_answer
                else "No direct governed answer is available."
            ),
        ),
        _checklist_item(
            "support_quality_checked",
            "Support-quality status was checked",
            "ready" if support_review.get("status_label") != "Blocked" else "blocked",
            f"Support quality status: {support_review.get('status_label', 'Unknown')}.",
        ),
        _checklist_item(
            "citations_present",
            "Citations are present",
            "ready" if citation_rows else "attention",
            (
                f"{len(citation_rows)} citations are attached."
                if citation_rows
                else "No citations are attached; reviewer should not treat this as complete."
            ),
        ),
        _checklist_item(
            "truth_context_checked",
            "Truth context was checked",
            "ready" if truth_present else "attention",
            (
                "At least one retrieved/cited chunk carries truth tags."
                if truth_present
                else "No truth-tagged chunk is visible in the current retrieval set."
            ),
        ),
        _checklist_item(
            "drift_context_checked",
            "Drift/caveat context was checked",
            "ready" if drift_present else "attention",
            (
                "Drift or caveat context is visible and separated from the direct answer."
                if drift_present
                else "No drift/caveat context is attached for this query."
            ),
        ),
        _checklist_item(
            "related_chunk_inspection",
            "Related chunk inspection is available",
            "ready"
            if source_detail_ready and source_detail.get("related_chunks")
            else "attention",
            (
                f"{len(source_detail.get('related_chunks', []))} related chunks are available."
                if source_detail_ready
                else "No source-detail target is ready for related chunk inspection."
            ),
        ),
        _checklist_item(
            "source_preview_checked",
            "Source preview eligibility was checked",
            "ready" if source_preview_checked and has_preview_signal else "attention",
            (
                f"Preview status: {source_detail.get('preview_status')}."
                if source_detail_ready
                else "Source preview eligibility was not evaluated for a selected chunk."
            ),
        ),
        _checklist_item(
            "export_ready",
            "Export packet is ready",
            "ready" if export_ready else "attention",
            (
                "Current governed review can be exported as markdown, text, or JSON."
                if export_ready
                else "Export packet has not been built or is blocked."
            ),
        ),
    ]

    ready_count = sum(1 for item in items if item["status"] == "ready")
    attention_count = sum(1 for item in items if item["status"] == "attention")
    blocked_count = sum(1 for item in items if item["status"] == "blocked")
    if blocked_count:
        status_key = "blocked"
        status_label = "Blocked"
    elif attention_count:
        status_key = "partial"
        status_label = "Review attention needed"
    else:
        status_key = "ready"
        status_label = "Review ready"

    notes: list[str] = []
    if coverage_status != "sufficient":
        notes.append(
            "Coverage is not marked sufficient; keep the review status partial until citations are inspected."
        )
    if not truth_present:
        notes.append("Canonical truth support is not visible in the retrieved set.")
    if not drift_present:
        notes.append(
            "No drift/caveat context is attached; this may be acceptable for some queries but should be checked."
        )
    if source_detail_ready and source_detail.get("preview_status") == "ineligible":
        notes.append(
            "Selected source detail is governed at chunk level, but whole-file preview is not eligible."
        )
    if export_ready:
        notes.append("Export packet is ready; final reviewer exports include checklist metadata.")
    if not notes:
        notes.append("Checklist is complete for the selected governed query.")

    return {
        "status_key": status_key,
        "status_label": status_label,
        "completeness_score": round(ready_count / len(items), 3),
        "items": items,
        "summary": {
            "total_items": len(items),
            "ready_items": ready_count,
            "attention_items": attention_count,
            "blocked_items": blocked_count,
            "coverage_status": coverage_status,
            "notes": notes,
        },
    }


def build_source_detail_options(answer_view: dict[str, Any]) -> list[dict[str, str]]:
    if answer_view["status"] != "ready":
        return []

    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_name, rows in (
        ("citation", answer_view["citation_rows"]),
        ("retrieval", answer_view["retrieval_rows"]),
    ):
        for row in rows:
            chunk_id = row["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            options.append(
                {
                    "chunk_id": chunk_id,
                    "label": f"{row['title']} ({chunk_id})",
                    "record_source": source_name,
                }
            )
    return options


def build_source_detail_view(
    answer_view: dict[str, Any],
    selected_chunk_id: str,
    *,
    related_limit: int = 5,
) -> dict[str, Any]:
    if answer_view["status"] != "ready":
        return {
            "status": "blocked",
            "browser_note": "Source detail is available only after a governed answer view is ready.",
        }

    rows_by_chunk: dict[str, dict[str, Any]] = {}
    for row in answer_view["retrieval_rows"]:
        rows_by_chunk[row["chunk_id"]] = {**row, "record_source": "retrieval"}
    for row in answer_view["citation_rows"]:
        rows_by_chunk[row["chunk_id"]] = {
            **rows_by_chunk.get(row["chunk_id"], {}),
            **row,
            "record_source": "citation",
        }

    selected = rows_by_chunk.get(selected_chunk_id)
    if selected is None:
        return {
            "status": "missing",
            "browser_note": f"No governed chunk record found for {selected_chunk_id!r}.",
        }

    chunk_records = _chunk_record_lookup()
    selected_chunk = chunk_records.get(selected_chunk_id, {})
    selected_document_id = selected.get("document_id") or selected_chunk.get("document_id")
    selected_sources = set(selected.get("source_paths", []))

    related_chunks: list[dict[str, Any]] = []
    for chunk in chunk_records.values():
        if chunk["chunk_id"] == selected_chunk_id:
            continue
        same_document = chunk.get("document_id") == selected_document_id
        source_overlap = bool(selected_sources.intersection(chunk.get("source_paths", [])))
        if not same_document and not source_overlap:
            continue
        related_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "document_id": chunk["document_id"],
                "chunk_kind": chunk["chunk_kind"],
                "relationship": "same document" if same_document else "shared source path",
                "source_paths": chunk["source_paths"],
                "truth_tags": chunk["truth_tags"],
                "drift_tags": chunk["drift_tags"],
                "phase_tags": chunk["phase_tags"],
                "retrieval_role": chunk["retrieval_role"],
                "authority_level": chunk["authority_level"],
                "text_preview": chunk["text"][:420],
            }
        )

    related_chunks.sort(
        key=lambda item: (
            0 if item["relationship"] == "same document" else 1,
            str(item["document_id"]),
            str(item["chunk_id"]),
        )
    )
    selected_source_paths = selected.get("source_paths", selected_chunk.get("source_paths", []))
    preview_options = [
        evaluate_source_preview(path)
        for path in selected_source_paths
    ]
    preview_status = (
        "eligible"
        if any(option["eligible"] for option in preview_options)
        else "ineligible"
    )

    detail = {
        "status": "ready",
        "browser_level": "governed_chunk_level",
        "browser_note": (
            "This is a governed chunk-level source browser. It exposes retrieval-pack context "
            "and related chunks, not an unrestricted full-file viewer."
        ),
        "selected": {
            "chunk_id": selected_chunk_id,
            "title": selected.get("title") or selected_chunk.get("title", ""),
            "document_id": selected_document_id,
            "chunk_kind": selected_chunk.get("chunk_kind", "unknown"),
            "source_paths": selected_source_paths,
            "registry_refs": selected.get("registry_refs", selected_chunk.get("registry_refs", [])),
            "truth_tags": selected.get("truth_tags", selected_chunk.get("truth_tags", [])),
            "drift_tags": selected.get("drift_tags", selected_chunk.get("drift_tags", [])),
            "phase_tags": selected.get("phase_tags", selected_chunk.get("phase_tags", [])),
            "page_routes": selected.get("page_routes", selected_chunk.get("page_routes", [])),
            "page_titles": selected.get("page_titles", []),
            "retrieval_role": selected.get("retrieval_role", selected_chunk.get("retrieval_role", "")),
            "authority_level": selected.get("authority_level", selected_chunk.get("authority_level", "")),
            "similarity_score": selected.get("similarity_score"),
            "text_preview": selected.get("text_preview") or selected_chunk.get("text", "")[:420],
            "full_text": selected.get("full_text") or selected_chunk.get("text", ""),
            "caveats": selected.get("caveats", selected_chunk.get("caveats", [])),
            "record_source": selected.get("record_source", "retrieval_pack"),
        },
        "related_chunks": related_chunks[:related_limit],
        "preview_options": preview_options,
        "preview_status": preview_status,
        "evidence_trace": {
            "query": answer_view["query"],
            "answer_title": answer_view["answer"]["answer_title"],
            "selected_chunk_id": selected_chunk_id,
            "document_id": selected_document_id,
            "source_paths": selected_source_paths,
            "registry_refs": selected.get("registry_refs", selected_chunk.get("registry_refs", [])),
            "related_chunk_count": min(len(related_chunks), related_limit),
            "preview_status": preview_status,
            "preview_rationale": (
                "At least one governed source path is eligible for controlled read-only preview."
                if preview_status == "eligible"
                else "No selected source path passed the governed preview eligibility rules."
            ),
            "browser_level": "governed_chunk_level",
        },
    }
    return detail


def build_audit_summary_export(
    answer_view: dict[str, Any],
    support_review: dict[str, Any],
    source_detail: dict[str, Any] | None = None,
    checklist: dict[str, Any] | None = None,
    eligible_source_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if answer_view["status"] != "ready":
        return {
            "status": "blocked",
            "message": answer_view.get("message", "Answer view is not ready."),
            "markdown": "",
            "text": "",
            "json": "{}",
            "filenames": {},
        }

    answer = answer_view["answer"]
    manifest = _retrieval_pack_manifest()
    citation_rows = answer_view["citation_rows"]
    source_detail_payload = source_detail if source_detail and source_detail.get("status") == "ready" else None
    eligible_source_summary = (
        eligible_source_index["summary"]
        if eligible_source_index and eligible_source_index.get("status") == "ready"
        else None
    )
    packet = {
        "packet_type": "governed_pace_navigator_review",
        "query": answer["query"],
        "answer_title": answer["answer_title"],
        "direct_answer": answer["direct_answer"],
        "assembly_status": answer["assembly_status"],
        "coverage_summary": answer["coverage_summary"],
        "support_quality": {
            "status_label": support_review["status_label"],
            "indicators": support_review["indicators"],
            "review_notes": support_review["review_notes"],
        },
        "supporting_points": answer["supporting_points"],
        "drift_and_caveats": answer["drift_and_caveats"],
        "recommended_pages": answer["recommended_pages"],
        "citations": [
            {
                "chunk_id": citation["chunk_id"],
                "title": citation["title"],
                "document_id": citation["document_id"],
                "retrieval_role": citation["retrieval_role"],
                "authority_level": citation["authority_level"],
                "source_paths": citation["source_paths"],
                "registry_refs": citation["registry_refs"],
                "truth_tags": citation["truth_tags"],
                "drift_tags": citation["drift_tags"],
                "phase_tags": citation["phase_tags"],
            }
            for citation in citation_rows
        ],
        "selected_source_detail": source_detail_payload["selected"] if source_detail_payload else None,
        "evidence_trace": source_detail_payload["evidence_trace"] if source_detail_payload else None,
        "source_preview_summary": (
            {
                "preview_status": source_detail_payload["preview_status"],
                "preview_options": [
                    {
                        "source_path": option["source_path"],
                        "eligible": option["eligible"],
                        "status": option["status"],
                        "reason": option["reason"],
                        "extension": option["extension"],
                        "file_size_bytes": option["file_size_bytes"],
                        "is_truncated": option["is_truncated"],
                        "limit_note": option["limit_note"],
                    }
                    for option in source_detail_payload["preview_options"]
                ],
                "preview_text_exported": False,
            }
            if source_detail_payload
            else None
        ),
        "audit_checklist": (
            {
                "status_key": checklist["status_key"],
                "status_label": checklist["status_label"],
                "completeness_score": checklist["completeness_score"],
                "summary": checklist["summary"],
                "items": checklist["items"],
            }
            if checklist
            else None
        ),
        "eligible_source_index_summary": (
            {
                "total_sources": eligible_source_summary["total_sources"],
                "eligible_sources": eligible_source_summary["eligible_sources"],
                "blocked_sources": eligible_source_summary["blocked_sources"],
                "governance_note": eligible_source_summary["governance_note"],
            }
            if eligible_source_summary
            else None
        ),
        "build_context": {
            "retrieval_pack_version": manifest.get("pack_version"),
            "retrieval_pack_chunk_count": manifest.get("chunk_count"),
            "retrieval_pack_document_count": manifest.get("document_count"),
            "generator_module": manifest.get("generator_module"),
        },
        "governance_note": (
            "This packet contains governed answer, review, citation, and chunk-level source context. "
            "It is not a model-generated free-form answer and does not include secrets."
        ),
    }

    recommended_pages = [
        f"{item['title']} (`/{item['route']}`): {item['reason']}"
        for item in answer["recommended_pages"]
    ]
    citations = [
        (
            f"`{item['chunk_id']}` | {item['title']} | role={item['retrieval_role']} | "
            f"authority={item['authority_level']} | sources={', '.join(item['source_paths'])}"
        )
        for item in citation_rows
    ]
    selected_source_lines: list[str] = []
    preview_lines: list[str] = []
    checklist_lines: list[str] = []
    source_index_lines: list[str] = []
    if source_detail_payload:
        selected = source_detail_payload["selected"]
        selected_source_lines = [
            f"selected chunk: `{selected['chunk_id']}`",
            f"document: `{selected['document_id']}`",
            f"source paths: {', '.join(selected['source_paths'])}",
            f"browser level: {source_detail_payload['browser_level']}",
        ]
        preview_lines = [
            (
                f"{option['source_path']} | eligible={option['eligible']} | "
                f"reason={option['reason']}"
            )
            for option in source_detail_payload["preview_options"]
        ]
    if checklist:
        checklist_lines = [
            (
                f"{item['label']} | status={item['status']} | "
                f"{item['detail']}"
            )
            for item in checklist["items"]
        ]
    if eligible_source_summary:
        source_index_lines = [
            f"total governed sources: {eligible_source_summary['total_sources']}",
            f"preview eligible: {eligible_source_summary['eligible_sources']}",
            f"blocked or unavailable: {eligible_source_summary['blocked_sources']}",
            eligible_source_summary["governance_note"],
        ]

    markdown = "\n\n".join(
        [
            f"# Governed PACE Navigator Review: {answer['answer_title']}",
            f"**Query:** {answer['query']}",
            f"**Assembly status:** `{answer['assembly_status']}`",
            f"**Coverage:** `{answer['coverage_summary']['status']}`",
            f"**Support quality:** {support_review['status_label']}",
            "## Direct Governed Answer\n" + answer["direct_answer"],
            "## Supporting Points\n" + _as_markdown_list(answer["supporting_points"]),
            "## Drift And Caveats\n" + _as_markdown_list(answer["drift_and_caveats"]),
            "## Review Notes\n" + _as_markdown_list(support_review["review_notes"]),
            "## Recommended Pages\n" + _as_markdown_list(recommended_pages),
            "## Citations\n" + _as_markdown_list(citations),
            "## Selected Source Detail\n" + _as_markdown_list(selected_source_lines),
            "## Preview Eligibility\n" + _as_markdown_list(preview_lines),
            "## Audit Checklist\n" + _as_markdown_list(checklist_lines),
            "## Eligible Source Index Summary\n" + _as_markdown_list(source_index_lines),
            (
                "## Build Context\n"
                f"- retrieval pack version: {manifest.get('pack_version')}\n"
                f"- retrieval pack chunks: {manifest.get('chunk_count')}\n"
                f"- retrieval pack documents: {manifest.get('document_count')}"
            ),
            "## Governance Note\n" + packet["governance_note"],
        ]
    )
    text = markdown.replace("# ", "").replace("## ", "")
    safe_query = "".join(
        char if char.isalnum() else "-"
        for char in str(answer["query"]).lower()
    ).strip("-")
    safe_query = "-".join(part for part in safe_query.split("-") if part)[:80]

    return {
        "status": "ready",
        "packet": packet,
        "markdown": markdown,
        "text": text,
        "json": json.dumps(packet, indent=2, sort_keys=True),
        "filenames": {
            "markdown": f"pace-review-{safe_query}.md",
            "text": f"pace-review-{safe_query}.txt",
            "json": f"pace-review-{safe_query}.json",
        },
    }


def _answer_support_row(
    answer_view: dict[str, Any],
    support_review: dict[str, Any],
    source_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if answer_view["status"] != "ready":
        return {
            "query": answer_view["query"],
            "status": answer_view["status"],
            "answer_title": "Blocked",
            "support_quality_status": support_review["status_label"],
            "assembly_status": "blocked",
            "coverage_status": "blocked",
            "canonical_truth_present": False,
            "drift_context_present": False,
            "page_route_support_present": False,
            "reference_only_count": 0,
            "citation_count": 0,
            "recommended_pages": [],
            "review_notes": support_review["review_notes"],
            "drift_and_caveats": [],
            "selected_source_detail_present": False,
            "message": answer_view.get("message", ""),
        }

    answer = answer_view["answer"]
    retrieval_rows = answer_view["retrieval_rows"]
    return {
        "query": answer["query"],
        "status": answer_view["status"],
        "answer_title": answer["answer_title"],
        "direct_answer": answer["direct_answer"],
        "support_quality_status": support_review["status_label"],
        "assembly_status": answer["assembly_status"],
        "coverage_status": answer["coverage_summary"]["status"],
        "canonical_truth_present": any(row["truth_tags"] for row in retrieval_rows),
        "drift_context_present": any(row["drift_tags"] for row in retrieval_rows),
        "page_route_support_present": any(row["page_routes"] for row in retrieval_rows),
        "reference_only_count": sum(
            1 for row in retrieval_rows if row["retrieval_role"] == "reference_only"
        ),
        "citation_count": len(answer_view["citation_rows"]),
        "recommended_pages": answer["recommended_pages"],
        "review_notes": support_review["review_notes"],
        "drift_and_caveats": answer["drift_and_caveats"],
        "selected_source_detail_present": bool(
            source_detail and source_detail.get("status") == "ready"
        ),
        "top_citation_chunk_id": (
            answer_view["citation_rows"][0]["chunk_id"]
            if answer_view["citation_rows"]
            else None
        ),
    }


def build_audit_workflow(
    selected_queries: list[str],
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    if not selected_queries:
        return {
            "status": "empty",
            "selected_queries": [],
            "items": [],
            "comparison_rows": [],
            "summary": {
                "total_queries": 0,
                "ready_queries": 0,
                "blocked_queries": 0,
                "checklist_ready_queries": [],
                "checklist_attention_queries": [],
                "checklist_blocked_queries": [],
                "strong_queries": [],
                "partial_or_attention_queries": [],
                "drift_heavy_queries": [],
                "reference_supported_queries": [],
                "workflow_status": "No queries selected",
                "review_notes": ["Select at least one fixed governed query to build the audit workflow."],
            },
        }

    items: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    eligible_source_index = build_eligible_source_index()
    for query in selected_queries:
        answer_view = build_governed_answer_view(query, top_k=top_k)
        support_review = build_support_quality_review(answer_view)
        source_options = build_source_detail_options(answer_view)
        source_detail = (
            build_source_detail_view(answer_view, source_options[0]["chunk_id"])
            if source_options
            else None
        )
        preliminary_export = build_audit_summary_export(answer_view, support_review, source_detail)
        checklist = build_audit_checklist(
            answer_view,
            support_review,
            source_detail,
            preliminary_export,
        )
        export_payload = build_audit_summary_export(
            answer_view,
            support_review,
            source_detail,
            checklist,
            eligible_source_index,
        )
        row = _answer_support_row(answer_view, support_review, source_detail)
        row["checklist_status"] = checklist["status_label"]
        row["checklist_score"] = checklist["completeness_score"]
        row["checklist_ready_items"] = checklist["summary"]["ready_items"]
        row["checklist_attention_items"] = checklist["summary"]["attention_items"]
        row["checklist_blocked_items"] = checklist["summary"]["blocked_items"]
        items.append(
            {
                "query": query,
                "answer_view": answer_view,
                "support_review": support_review,
                "source_detail": source_detail,
                "audit_checklist": checklist,
                "single_query_export": export_payload,
                "comparison_row": row,
            }
        )
        comparison_rows.append(row)

    ready_count = sum(1 for row in comparison_rows if row["status"] == "ready")
    blocked_count = sum(1 for row in comparison_rows if row["status"] != "ready")
    strong_queries = [
        row["query"]
        for row in comparison_rows
        if row["support_quality_status"] == "Strong governed support"
    ]
    partial_or_attention_queries = [
        row["query"]
        for row in comparison_rows
        if row["support_quality_status"]
        in {"Partial governed support", "Needs reviewer attention", "Blocked"}
    ]
    drift_heavy_queries = [
        row["query"]
        for row in comparison_rows
        if row["drift_context_present"] or row["drift_and_caveats"]
    ]
    reference_supported_queries = [
        row["query"]
        for row in comparison_rows
        if int(row["reference_only_count"]) > 0
    ]
    checklist_ready_queries = [
        row["query"]
        for row in comparison_rows
        if row.get("checklist_status") == "Review ready"
    ]
    checklist_attention_queries = [
        row["query"]
        for row in comparison_rows
        if row.get("checklist_status") == "Review attention needed"
    ]
    checklist_blocked_queries = [
        row["query"]
        for row in comparison_rows
        if row.get("checklist_status") == "Blocked"
    ]

    if blocked_count == len(comparison_rows):
        workflow_status = "Blocked"
    elif blocked_count:
        workflow_status = "Mixed readiness"
    elif partial_or_attention_queries:
        workflow_status = "Ready with review attention"
    else:
        workflow_status = "Ready"

    review_notes: list[str] = []
    if strong_queries:
        review_notes.append(
            f"{len(strong_queries)} selected queries have strong governed support."
        )
    if partial_or_attention_queries:
        review_notes.append(
            f"{len(partial_or_attention_queries)} selected queries need closer reviewer attention."
        )
    if drift_heavy_queries:
        review_notes.append(
            f"{len(drift_heavy_queries)} selected queries include drift or caveat context that should be preserved."
        )
    if reference_supported_queries:
        review_notes.append(
            f"{len(reference_supported_queries)} selected queries use reference-only chunks; review citations before overclaiming."
        )
    if checklist_attention_queries:
        review_notes.append(
            f"{len(checklist_attention_queries)} selected queries have checklist attention items before export."
        )
    if checklist_blocked_queries:
        review_notes.append(
            f"{len(checklist_blocked_queries)} selected queries have blocked checklist items."
        )
    if blocked_count:
        review_notes.append(
            f"{blocked_count} selected queries are blocked by retrieval runtime setup."
        )
    if not review_notes:
        review_notes.append("All selected queries are ready with no extra workflow warnings.")

    return {
        "status": "ready" if ready_count else "blocked",
        "selected_queries": selected_queries,
        "items": items,
        "comparison_rows": comparison_rows,
        "summary": {
            "total_queries": len(comparison_rows),
            "ready_queries": ready_count,
            "blocked_queries": blocked_count,
            "checklist_ready_queries": checklist_ready_queries,
            "checklist_attention_queries": checklist_attention_queries,
            "checklist_blocked_queries": checklist_blocked_queries,
            "strong_queries": strong_queries,
            "partial_or_attention_queries": partial_or_attention_queries,
            "drift_heavy_queries": drift_heavy_queries,
            "reference_supported_queries": reference_supported_queries,
            "workflow_status": workflow_status,
            "review_notes": review_notes,
        },
    }


def build_cross_query_audit_export(workflow: dict[str, Any]) -> dict[str, Any]:
    if workflow["status"] == "empty":
        return {
            "status": "blocked",
            "message": "No selected queries are available for combined export.",
            "markdown": "",
            "text": "",
            "json": "{}",
            "filenames": {},
        }

    manifest = _retrieval_pack_manifest()
    summary = workflow["summary"]
    eligible_source_summary = build_eligible_source_index()["summary"]
    packet_items = []
    for item in workflow["items"]:
        row = item["comparison_row"]
        answer_view = item["answer_view"]
        answer = answer_view.get("answer", {})
        source_detail = item.get("source_detail")
        source_detail_ready = source_detail and source_detail.get("status") == "ready"
        checklist = item.get("audit_checklist")
        packet_items.append(
            {
                "query": row["query"],
                "status": row["status"],
                "answer_title": row["answer_title"],
                "direct_answer": row.get("direct_answer", ""),
                "support_quality_status": row["support_quality_status"],
                "assembly_status": row["assembly_status"],
                "coverage_status": row["coverage_status"],
                "canonical_truth_present": row["canonical_truth_present"],
                "drift_context_present": row["drift_context_present"],
                "page_route_support_present": row["page_route_support_present"],
                "reference_only_count": row["reference_only_count"],
                "citation_count": row["citation_count"],
                "audit_checklist": (
                    {
                        "status_key": checklist["status_key"],
                        "status_label": checklist["status_label"],
                        "completeness_score": checklist["completeness_score"],
                        "summary": checklist["summary"],
                        "items": checklist["items"],
                    }
                    if checklist
                    else None
                ),
                "supporting_points": answer.get("supporting_points", []),
                "review_notes": row["review_notes"],
                "drift_and_caveats": row["drift_and_caveats"],
                "recommended_pages": row["recommended_pages"],
                "citations": [
                    {
                        "chunk_id": citation["chunk_id"],
                        "title": citation["title"],
                        "document_id": citation["document_id"],
                        "retrieval_role": citation["retrieval_role"],
                        "authority_level": citation["authority_level"],
                        "source_paths": citation["source_paths"],
                        "truth_tags": citation["truth_tags"],
                        "drift_tags": citation["drift_tags"],
                        "phase_tags": citation["phase_tags"],
                    }
                    for citation in answer_view.get("citation_rows", [])
                ],
                "evidence_trace": source_detail["evidence_trace"] if source_detail_ready else None,
                "source_preview_summary": (
                    {
                        "preview_status": source_detail["preview_status"],
                        "preview_options": [
                            {
                                "source_path": option["source_path"],
                                "eligible": option["eligible"],
                                "reason": option["reason"],
                                "extension": option["extension"],
                                "file_size_bytes": option["file_size_bytes"],
                            }
                            for option in source_detail["preview_options"]
                        ],
                        "preview_text_exported": False,
                    }
                    if source_detail_ready
                    else None
                ),
            }
        )

    packet = {
        "packet_type": "governed_pace_navigator_cross_query_audit",
        "selected_queries": workflow["selected_queries"],
        "workflow_summary": summary,
        "query_reviews": packet_items,
        "eligible_source_index_summary": {
            "total_sources": eligible_source_summary["total_sources"],
            "eligible_sources": eligible_source_summary["eligible_sources"],
            "blocked_sources": eligible_source_summary["blocked_sources"],
            "governance_note": eligible_source_summary["governance_note"],
        },
        "build_context": {
            "retrieval_pack_version": manifest.get("pack_version"),
            "retrieval_pack_chunk_count": manifest.get("chunk_count"),
            "retrieval_pack_document_count": manifest.get("document_count"),
            "generator_module": manifest.get("generator_module"),
        },
        "governance_note": (
            "This packet compares fixed governed queries only. It is deterministic, citation-backed, "
            "and does not contain secrets or generated free-form analysis."
        ),
    }

    comparison_lines = [
        (
            f"{row['query']} | support={row['support_quality_status']} | "
            f"truth={row['canonical_truth_present']} | drift={row['drift_context_present']} | "
            f"citations={row['citation_count']} | reference_only={row['reference_only_count']} | "
            f"checklist={row.get('checklist_status', 'Not checked')}"
        )
        for row in workflow["comparison_rows"]
    ]
    per_query_sections = []
    for item in packet_items:
        recommended_pages = [
            f"{page['title']} (`/{page['route']}`): {page['reason']}"
            for page in item["recommended_pages"]
        ]
        citations = [
            f"`{citation['chunk_id']}` | {citation['title']} | sources={', '.join(citation['source_paths'])}"
            for citation in item["citations"]
        ]
        preview_lines = []
        checklist_lines = []
        if item.get("source_preview_summary"):
            preview_lines = [
                f"{option['source_path']} | eligible={option['eligible']} | reason={option['reason']}"
                for option in item["source_preview_summary"]["preview_options"]
            ]
        if item.get("audit_checklist"):
            checklist_lines = [
                f"{entry['label']} | status={entry['status']} | {entry['detail']}"
                for entry in item["audit_checklist"]["items"]
            ]
        per_query_sections.append(
            "\n\n".join(
                [
                    f"### {item['answer_title']}",
                    f"**Query:** {item['query']}",
                    f"**Support quality:** {item['support_quality_status']}",
                    f"**Coverage:** `{item['coverage_status']}`",
                    "Direct answer:\n" + (item["direct_answer"] or "Blocked or unavailable."),
                    "Review notes:\n" + _as_markdown_list(item["review_notes"]),
                    "Drift and caveats:\n" + _as_markdown_list(item["drift_and_caveats"]),
                    "Recommended pages:\n" + _as_markdown_list(recommended_pages),
                    "Citations:\n" + _as_markdown_list(citations),
                    "Preview eligibility:\n" + _as_markdown_list(preview_lines),
                    "Audit checklist:\n" + _as_markdown_list(checklist_lines),
                ]
            )
        )

    markdown = "\n\n".join(
        [
            "# Governed PACE Navigator Cross-Query Audit",
            f"**Workflow status:** {summary['workflow_status']}",
            f"**Selected queries:** {summary['total_queries']}",
            f"**Ready queries:** {summary['ready_queries']}",
            f"**Blocked queries:** {summary['blocked_queries']}",
            "## Workflow Review Notes\n" + _as_markdown_list(summary["review_notes"]),
            "## Comparison Matrix\n" + _as_markdown_list(comparison_lines),
            (
                "## Eligible Source Index Summary\n"
                f"- total governed sources: {eligible_source_summary['total_sources']}\n"
                f"- preview eligible: {eligible_source_summary['eligible_sources']}\n"
                f"- blocked or unavailable: {eligible_source_summary['blocked_sources']}\n"
                f"- {eligible_source_summary['governance_note']}"
            ),
            "## Per-Query Reviews\n" + "\n\n".join(per_query_sections),
            (
                "## Build Context\n"
                f"- retrieval pack version: {manifest.get('pack_version')}\n"
                f"- retrieval pack chunks: {manifest.get('chunk_count')}\n"
                f"- retrieval pack documents: {manifest.get('document_count')}"
            ),
            "## Governance Note\n" + packet["governance_note"],
        ]
    )
    text = markdown.replace("# ", "").replace("## ", "").replace("### ", "")
    query_count = summary["total_queries"]
    return {
        "status": "ready",
        "packet": packet,
        "markdown": markdown,
        "text": text,
        "json": json.dumps(packet, indent=2, sort_keys=True),
        "filenames": {
            "markdown": f"pace-cross-query-audit-{query_count}-queries.md",
            "text": f"pace-cross-query-audit-{query_count}-queries.txt",
            "json": f"pace-cross-query-audit-{query_count}-queries.json",
        },
    }
