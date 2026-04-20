from __future__ import annotations

from typing import Any

from app.services import (
    get_drift_items,
    get_pace_phase,
    get_runtime_governance_summary,
    get_truth_entries,
    recommend_page_for_topic,
)

_PHASE_ORDER = ("plan", "analyze", "construct", "execute")
_DEFAULT_TOPIC_PREVIEWS = (
    "where the public model truth lives",
    "team exposure prioritization for managers",
    "threshold tradeoffs and confusion matrix",
    "why fallback is not the final model probability",
)


def _first_truth_entry(domain: str) -> dict[str, Any]:
    matches = get_truth_entries(domain)
    if not matches:
        raise KeyError(f"No truth entry found for domain {domain!r}.")
    return matches[0]


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
        "pace_spine_note": method_truth["description"],
    }
