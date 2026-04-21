from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.services import (
    NavigatorEmbeddingConfigurationError,
    NavigatorEmbeddingIndexError,
    NavigatorEmbeddingRequestError,
    NavigatorRetrievalIndexNotFoundError,
    OpenAIEmbeddingConfig,
    assemble_governed_answer,
    get_drift_items,
    get_pace_phase,
    get_retrieval_evaluation_queries,
    get_runtime_governance_summary,
    get_truth_entries,
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
            "source_paths": selected.get("source_paths", selected_chunk.get("source_paths", [])),
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
    }
    return detail


def build_audit_summary_export(
    answer_view: dict[str, Any],
    support_review: dict[str, Any],
    source_detail: dict[str, Any] | None = None,
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
    if source_detail_payload:
        selected = source_detail_payload["selected"]
        selected_source_lines = [
            f"selected chunk: `{selected['chunk_id']}`",
            f"document: `{selected['document_id']}`",
            f"source paths: {', '.join(selected['source_paths'])}",
            f"browser level: {source_detail_payload['browser_level']}",
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
