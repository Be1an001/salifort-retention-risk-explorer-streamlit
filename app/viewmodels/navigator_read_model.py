from __future__ import annotations

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
                "caveats": item["caveats"],
            }
        )

    citation_rows = []
    for citation in answer["citations"]:
        citation_rows.append(
            {
                "chunk_id": citation["chunk_id"],
                "document_id": citation["document_id"],
                "title": citation["title"],
                "source_paths": citation["source_paths"],
                "registry_refs": citation["registry_refs"],
                "truth_tags": citation["truth_tags"],
                "drift_tags": citation["drift_tags"],
                "phase_tags": citation["phase_tags"],
                "retrieval_role": citation["retrieval_role"],
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
