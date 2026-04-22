from __future__ import annotations

import re
from typing import Any

from app.services.navigator_loader import load_all_navigator_registries

_PAGE_ROUTES = {
    "Overview": "overview",
    "Workforce Explorer": "workforce-explorer",
    "EDA & Patterns": "eda-patterns",
    "Model & Threshold Lab": "model-threshold-lab",
    "Explainability": "explainability",
    "Manager Action View": "manager-action-view",
    "Methods & Limitations": "methods-limitations",
}

_PAGE_KEYWORDS = {
    "Overview": {
        "overview",
        "summary",
        "question",
        "identity",
        "project",
        "kpi",
        "headline",
        "intro",
    },
    "Workforce Explorer": {
        "employee",
        "employees",
        "workforce",
        "filter",
        "screening",
        "department",
        "row",
        "row-level",
        "explorer",
        "headcount",
        "attrition probability",
    },
    "EDA & Patterns": {
        "eda",
        "pattern",
        "patterns",
        "hours",
        "satisfaction",
        "salary",
        "tenure",
        "heatmap",
        "workload",
        "promotion",
        "survival",
    },
    "Model & Threshold Lab": {
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "accuracy",
        "confusion",
        "validation",
        "pr curve",
        "xgboost",
    },
    "Explainability": {
        "shap",
        "explainability",
        "explanation",
        "feature importance",
        "driver",
        "drivers",
        "interpretation",
        "dependence",
    },
    "Manager Action View": {
        "manager",
        "action",
        "intervention",
        "exposure",
        "prioritize",
        "queue",
        "review",
        "team burden",
    },
    "Methods & Limitations": {
        "method",
        "methods",
        "limitations",
        "governance",
        "fallback",
        "artifact-backed",
        "runtime",
        "scope",
        "assumption",
    },
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_truth_entries(domain: str | None = None) -> list[dict[str, Any]]:
    truth_domains = load_all_navigator_registries().truth_registry["truth_domains"]
    entries = []
    for domain_name, entry in truth_domains.items():
        record = {"domain": domain_name, **entry}
        entries.append(record)

    if domain is None:
        return entries

    normalized_domain = _normalize_token(domain)
    return [
        entry
        for entry in entries
        if _normalize_token(entry["domain"]) == normalized_domain
        or _normalize_token(entry["truth_id"]) == normalized_domain
    ]


def get_drift_items(
    severity: str | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    drift_items = list(load_all_navigator_registries().drift_register["drifts"])

    if severity is not None:
        severity_normalized = _normalize_token(severity)
        drift_items = [
            item
            for item in drift_items
            if _normalize_token(str(item["severity"])) == severity_normalized
        ]

    if status is not None:
        status_normalized = _normalize_token(status)
        drift_items = [
            item
            for item in drift_items
            if _normalize_token(str(item["status"])) == status_normalized
        ]

    return drift_items


def get_pace_phase(phase_name: str) -> dict[str, Any]:
    normalized_phase = _normalize_token(phase_name)
    for phase in load_all_navigator_registries().pace_phase_map["phases"]:
        if _normalize_token(str(phase["phase_id"])) == normalized_phase or _normalize_token(
            str(phase["phase_title"])
        ) == normalized_phase:
            return {
                "normalized_phase_name": str(phase["phase_id"]).lower(),
                **phase,
            }
    raise KeyError(f"PACE phase not found for {phase_name!r}.")


def lookup_glossary(term: str) -> dict[str, Any] | None:
    entries = load_all_navigator_registries().glossary["terms"]
    normalized_query = _normalize_token(term)

    if not normalized_query:
        return None

    for entry in entries:
        normalized_term = _normalize_token(entry["term"])
        if normalized_term == normalized_query:
            return {"match_type": "exact", **entry}

    for entry in entries:
        normalized_term = _normalize_token(entry["term"])
        if normalized_query in normalized_term or normalized_term in normalized_query:
            return {"match_type": "normalized_alias", **entry}

    return None


def _build_page_to_phase_map() -> dict[str, str]:
    page_to_phase: dict[str, str] = {}
    for phase in load_all_navigator_registries().pace_phase_map["phases"]:
        phase_name = str(phase["phase_id"]).lower()
        for page_name in phase["app_pages"]:
            page_to_phase.setdefault(page_name, phase_name)
    return page_to_phase


def _build_page_to_source_ids() -> dict[str, list[str]]:
    page_to_source_ids: dict[str, list[str]] = {}
    for source in load_all_navigator_registries().source_registry["sources"]:
        for page_name in source["consumer_pages"]:
            page_to_source_ids.setdefault(page_name, []).append(source["source_id"])
    return page_to_source_ids


def recommend_page_for_topic(topic: str) -> dict[str, Any]:
    normalized_topic = _normalize_text(topic)
    page_scores = {page_name: 0 for page_name in _PAGE_ROUTES}
    matched_terms: dict[str, list[str]] = {page_name: [] for page_name in _PAGE_ROUTES}

    for page_name, keywords in _PAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_topic:
                page_scores[page_name] += max(1, len(keyword.split()))
                matched_terms[page_name].append(keyword)

    glossary_match = lookup_glossary(topic)
    if glossary_match is not None:
        glossary_term = str(glossary_match["term"])
        glossary_text = _normalize_text(glossary_term)
        for page_name, keywords in _PAGE_KEYWORDS.items():
            if any(keyword in glossary_text or glossary_text in keyword for keyword in keywords):
                page_scores[page_name] += 2
                matched_terms[page_name].append(f"glossary:{glossary_term}")

    best_page = max(
        page_scores,
        key=lambda page_name: (page_scores[page_name], -list(_PAGE_ROUTES).index(page_name)),
    )
    if page_scores[best_page] == 0:
        best_page = "Overview"
        matched_terms[best_page].append("default")

    page_to_phase = _build_page_to_phase_map()
    page_to_source_ids = _build_page_to_source_ids()
    phase_name = page_to_phase.get(best_page, "execute")

    reason = (
        f"Matched topic cues {matched_terms[best_page]} to {best_page}. "
        f"In the current page map, that destination sits under the "
        f"{phase_name.capitalize()} phase; related topics may also connect to other phases."
    )

    return {
        "topic": topic,
        "recommended_page_title": best_page,
        "recommended_page_route": _PAGE_ROUTES[best_page],
        "reason": reason,
        "supporting_phase": phase_name,
        "related_source_ids": page_to_source_ids.get(best_page, []),
        "matched_terms": matched_terms[best_page],
    }


def get_runtime_governance_summary() -> dict[str, Any]:
    truth_entries = {entry["domain"]: entry for entry in get_truth_entries()}
    high_priority_drifts = [
        {
            "drift_id": item["drift_id"],
            "title": item["title"],
            "severity": item["severity"],
            "status": item["status"],
        }
        for item in get_drift_items()
        if item["drift_id"]
        in {
            "drift_public_selection_vs_rerun_leader",
            "drift_runtime_rows_vs_fallback_rows",
            "drift_builder_reference_vs_local_reimplementation",
        }
    ]

    return {
        "public_model_truth": {
            "truth_id": truth_entries["public_model_truth"]["truth_id"],
            "title": truth_entries["public_model_truth"]["title"],
            "description": truth_entries["public_model_truth"]["description"],
            "authority_rule": truth_entries["public_model_truth"]["authority_rule"],
        },
        "artifact_backed_runtime_truth": {
            "truth_id": truth_entries["runtime_truth"]["truth_id"],
            "title": truth_entries["runtime_truth"]["title"],
            "description": truth_entries["runtime_truth"]["description"],
            "authority_rule": truth_entries["runtime_truth"]["authority_rule"],
        },
        "fallback_truth": {
            "truth_id": truth_entries["fallback_truth"]["truth_id"],
            "title": truth_entries["fallback_truth"]["title"],
            "description": truth_entries["fallback_truth"]["description"],
            "authority_rule": truth_entries["fallback_truth"]["authority_rule"],
        },
        "important_drift_notes": high_priority_drifts,
    }
