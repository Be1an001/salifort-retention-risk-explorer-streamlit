from __future__ import annotations

import json
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from app.services.navigator_loader import get_repo_root, load_all_navigator_registries
from app.services.navigator_queries import get_drift_items, get_pace_phase, get_truth_entries

_PAGE_NAME_TO_ROUTE = {
    "Overview": "overview",
    "PACE Navigator": "pace-navigator",
    "Workforce Explorer": "workforce-explorer",
    "EDA & Patterns": "eda-patterns",
    "Model & Threshold Lab": "model-threshold-lab",
    "Explainability": "explainability",
    "Manager Action View": "manager-action-view",
    "Methods & Limitations": "methods-limitations",
}

_MARKDOWN_SOURCES = [
    ("README.md", "answer_ready"),
    ("artifacts/v2/README.md", "answer_ready"),
    ("navigator/README.md", "answer_ready"),
    ("navigator/repo_rename_assessment.md", "reference_only"),
    ("docs/navigator/wp2-readonly-consumption-notes.md", "reference_only"),
    ("docs/navigator/wp3-page-shell-notes.md", "reference_only"),
    ("docs/navigator/wp4-governed-interactions-notes.md", "reference_only"),
    ("docs/navigator/wp5-retrieval-preparation-notes.md", "reference_only"),
]

_SMALL_JSON_SOURCES = [
    ("artifacts/v2/metadata.json", "answer_ready"),
    ("artifacts/v2/model_modes_summary.json", "reference_only"),
    ("artifacts/v2/schemas/artifact_contract.json", "answer_ready"),
]

_EXCLUDED_CONTENT = [
    {
        "category": "raw_row_data",
        "paths": [
            "data/hr_capstone_dataset.csv",
            "artifacts/v2/employee_scores.parquet",
            "artifacts/v2/employee_shap_sample.parquet",
            "artifacts/v2/pr_curve_points.parquet",
        ],
        "reason": "Row-level or large raw data should not be flattened into answer-ready retrieval text.",
    },
    {
        "category": "artifact_table_dumps",
        "paths": [
            "artifacts/v2/department_exposure.csv",
            "artifacts/v2/threshold_curve.csv",
            "artifacts/v2/validation_model_comparison.csv",
            "artifacts/v2/confusion_matrix_at_selected_threshold.csv",
            "artifacts/v2/shap_importance.csv",
            "artifacts/v2/metadata.template.json",
        ],
        "reason": "Structured artifact tables and draft templates are represented by governed summaries or registry/source entries instead of raw row dumps.",
    },
    {
        "category": "binary_assets",
        "paths": [
            "outputs/figures/",
        ],
        "reason": "Binary figures are not turned into retrieval text in WP5.",
    },
    {
        "category": "raw_code_bodies",
        "paths": [
            "app/pages/*.py",
            "app/utils/*.py",
            "scripts/build_v2_artifacts.py",
        ],
        "reason": "WP5 uses source inventory summaries and documentation, not raw code-body dumps, for governed retrieval preparation.",
    },
]

_SPECIAL_HANDLING = [
    {
        "category": "external_legacy_reference",
        "paths": [
            "../salifort-motors-attrition-modeling-python/README.md",
            "../salifort-motors-attrition-modeling-python/scripts/02_salifort_motors_capstone_portfolio_project.py",
        ],
        "handling": "Represented as stub source entries only. Content is not ingested because it is not locally readable in this repo.",
    },
    {
        "category": "machine_specific_provenance",
        "paths": [
            "artifacts/v2/metadata.json",
            "artifacts/v2/model_modes_summary.json",
        ],
        "handling": "Included with caveats because the files contain machine-specific path provenance that should be preserved but not over-generalized.",
    },
]


def get_retrieval_pack_root() -> Path:
    return get_repo_root() / "navigator" / "retrieval_pack"


def _slugify(value: str) -> str:
    slug_chars = []
    previous_was_separator = False
    for char in value.lower():
        if char.isalnum():
            slug_chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            slug_chars.append("-")
            previous_was_separator = True
    return "".join(slug_chars).strip("-") or "root"


def _page_routes_from_names(page_names: list[str]) -> list[str]:
    routes = []
    for page_name in page_names:
        route = _PAGE_NAME_TO_ROUTE.get(page_name)
        if route is not None and route not in routes:
            routes.append(route)
    return routes


def _path_matches_rule(path: str, rule: str) -> bool:
    normalized_path = path.replace("\\", "/")
    normalized_rule = rule.replace("\\", "/")
    if "*" in normalized_rule or "?" in normalized_rule:
        return fnmatch(normalized_path, normalized_rule)
    if normalized_rule.endswith("/"):
        return normalized_path.startswith(normalized_rule)
    return normalized_path == normalized_rule


def _get_exclusion_matches(path: str) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for item in _EXCLUDED_CONTENT:
        for rule in item["paths"]:
            if _path_matches_rule(path, rule):
                matches.append(
                    {
                        "category": item["category"],
                        "reason": item["reason"],
                    }
                )
                break
    return matches


def _get_special_handling_matches(path: str) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for item in _SPECIAL_HANDLING:
        for rule in item["paths"]:
            if _path_matches_rule(path, rule):
                matches.append(
                    {
                        "category": item["category"],
                        "handling": item["handling"],
                    }
                )
                break
    return matches


def _iter_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.lstrip("#").strip() or "Untitled Section"
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if body]


def _append_document(
    documents: list[dict[str, Any]],
    *,
    document_id: str,
    document_class: str,
    title: str,
    content: str,
    source_paths: list[str],
    registry_refs: list[str],
    truth_tags: list[str],
    drift_tags: list[str],
    phase_tags: list[str],
    page_routes: list[str],
    authority_level: str,
    retrieval_role: str,
    caveats: list[str] | None = None,
) -> None:
    documents.append(
        {
            "document_id": document_id,
            "document_class": document_class,
            "title": title,
            "content": content.strip(),
            "source_paths": source_paths,
            "registry_refs": registry_refs,
            "truth_tags": truth_tags,
            "drift_tags": drift_tags,
            "phase_tags": phase_tags,
            "page_routes": page_routes,
            "authority_level": authority_level,
            "retrieval_role": retrieval_role,
            "caveats": caveats or [],
        }
    )


def _append_chunk(
    chunks: list[dict[str, Any]],
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    chunk_kind: str,
    title: str,
    text: str,
    source_paths: list[str],
    registry_refs: list[str],
    truth_tags: list[str],
    drift_tags: list[str],
    phase_tags: list[str],
    page_routes: list[str],
    authority_level: str,
    retrieval_role: str,
    caveats: list[str] | None = None,
) -> None:
    chunks.append(
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "chunk_kind": chunk_kind,
            "title": title,
            "text": text.strip(),
            "source_paths": source_paths,
            "registry_refs": registry_refs,
            "truth_tags": truth_tags,
            "drift_tags": drift_tags,
            "phase_tags": phase_tags,
            "page_routes": page_routes,
            "authority_level": authority_level,
            "retrieval_role": retrieval_role,
            "caveats": caveats or [],
        }
    )


def _build_truth_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    for entry in get_truth_entries():
        document_id = f"truth::{entry['truth_id']}"
        content = (
            f"{entry['title']}\n\n"
            f"Description: {entry['description']}\n\n"
            f"Authority rule: {entry['authority_rule']}\n\n"
            f"Preserve in upgrade: {'yes' if entry['preserve_in_upgrade'] else 'no'}"
        )
        source_paths = list(dict.fromkeys(entry["primary_sources"] + entry["secondary_sources"]))
        caveats = []
        if any(path.startswith("../") for path in source_paths):
            caveats.append("Contains external legacy reference paths that are not locally readable in this repo.")

        _append_document(
            documents,
            document_id=document_id,
            document_class="truth_entry",
            title=entry["title"],
            content=content,
            source_paths=source_paths,
            registry_refs=[document_id],
            truth_tags=[entry["domain"]],
            drift_tags=[],
            phase_tags=[],
            page_routes=[],
            authority_level="canonical_truth",
            retrieval_role="answer_ready",
            caveats=caveats,
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::summary",
            document_id=document_id,
            chunk_index=0,
            chunk_kind="truth_summary",
            title=f"{entry['title']} summary",
            text=content,
            source_paths=source_paths,
            registry_refs=[document_id],
            truth_tags=[entry["domain"]],
            drift_tags=[],
            phase_tags=[],
            page_routes=[],
            authority_level="canonical_truth",
            retrieval_role="answer_ready",
            caveats=caveats,
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::sources",
            document_id=document_id,
            chunk_index=1,
            chunk_kind="truth_sources",
            title=f"{entry['title']} sources",
            text=(
                "Primary sources:\n- "
                + "\n- ".join(entry["primary_sources"])
                + "\n\nSecondary sources:\n- "
                + "\n- ".join(entry["secondary_sources"])
            ),
            source_paths=source_paths,
            registry_refs=[document_id],
            truth_tags=[entry["domain"]],
            drift_tags=[],
            phase_tags=[],
            page_routes=[],
            authority_level="canonical_truth",
            retrieval_role="answer_ready",
            caveats=caveats,
        )


def _build_drift_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    for item in get_drift_items():
        document_id = f"drift::{item['drift_id']}"
        content = (
            f"{item['title']}\n\n"
            f"Severity: {item['severity']}\n"
            f"Status: {item['status']}\n\n"
            f"Canonical side: {item['canonical_side']}\n\n"
            f"Current side: {item['current_side']}\n\n"
            f"User-visible risk: {item['user_visible_risk']}"
        )
        _append_document(
            documents,
            document_id=document_id,
            document_class="drift_item",
            title=item["title"],
            content=content,
            source_paths=item["source_evidence"],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[item["drift_id"]],
            phase_tags=[],
            page_routes=[],
            authority_level="governed_drift",
            retrieval_role="answer_ready",
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::summary",
            document_id=document_id,
            chunk_index=0,
            chunk_kind="drift_summary",
            title=f"{item['title']} summary",
            text=content,
            source_paths=item["source_evidence"],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[item["drift_id"]],
            phase_tags=[],
            page_routes=[],
            authority_level="governed_drift",
            retrieval_role="answer_ready",
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::handling",
            document_id=document_id,
            chunk_index=1,
            chunk_kind="drift_handling",
            title=f"{item['title']} handling",
            text=(
                f"Handling rule: {item['upgrade_handling_rule']}\n\nNotes:\n- "
                + "\n- ".join(item["notes"])
            ),
            source_paths=item["source_evidence"],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[item["drift_id"]],
            phase_tags=[],
            page_routes=[],
            authority_level="governed_drift",
            retrieval_role="answer_ready",
        )


def _build_phase_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    for phase_name in ("plan", "analyze", "construct", "execute"):
        phase = get_pace_phase(phase_name)
        document_id = f"phase::{phase['normalized_phase_name']}"
        page_routes = _page_routes_from_names(phase["app_pages"])
        source_paths = phase["new_repo_runtime_links"] + phase["old_repo_method_links"]
        _append_document(
            documents,
            document_id=document_id,
            document_class="phase_entry",
            title=phase["phase_title"],
            content=(
                f"{phase['phase_title']}\n\n"
                f"Goal: {phase['phase_goal']}\n\n"
                f"Portfolio meaning: {phase['portfolio_meaning']}"
            ),
            source_paths=source_paths,
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=phase["known_drifts"],
            phase_tags=[phase["normalized_phase_name"]],
            page_routes=page_routes,
            authority_level="navigation_spine",
            retrieval_role="answer_ready",
            caveats=[
                "Old repo method links are external legacy references when they point outside this repo."
            ]
            if phase["old_repo_method_links"]
            else [],
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::overview",
            document_id=document_id,
            chunk_index=0,
            chunk_kind="phase_overview",
            title=f"{phase['phase_title']} overview",
            text=(
                f"Goal: {phase['phase_goal']}\n\n"
                f"Portfolio meaning: {phase['portfolio_meaning']}\n\n"
                f"Key takeaways:\n- " + "\n- ".join(phase["key_takeaways"])
            ),
            source_paths=source_paths,
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=phase["known_drifts"],
            phase_tags=[phase["normalized_phase_name"]],
            page_routes=page_routes,
            authority_level="navigation_spine",
            retrieval_role="answer_ready",
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::assets",
            document_id=document_id,
            chunk_index=1,
            chunk_kind="phase_assets",
            title=f"{phase['phase_title']} assets",
            text=(
                "App pages:\n- "
                + "\n- ".join(phase["app_pages"])
                + "\n\nArtifacts:\n- "
                + "\n- ".join(phase["artifacts"])
                + "\n\nFigures:\n- "
                + "\n- ".join(phase["figures"])
                + "\n\nFuture navigator use: "
                + phase["future_navigator_use"]
            ),
            source_paths=source_paths,
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=phase["known_drifts"],
            phase_tags=[phase["normalized_phase_name"]],
            page_routes=page_routes,
            authority_level="navigation_spine",
            retrieval_role="answer_ready",
        )


def _build_glossary_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    glossary_terms = load_all_navigator_registries().glossary["terms"]
    for entry in glossary_terms:
        term_slug = _slugify(entry["term"])
        document_id = f"glossary::{term_slug}"
        text = (
            f"Definition: {entry['definition']}\n\n"
            f"Preferred usage: {entry['preferred_usage']}\n\n"
            f"Avoid confusion with:\n- " + "\n- ".join(entry["avoid_confusion_with"])
        )
        _append_document(
            documents,
            document_id=document_id,
            document_class="glossary_term",
            title=entry["term"],
            content=text,
            source_paths=entry["related_sources"],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[],
            phase_tags=[],
            page_routes=[],
            authority_level="terminology",
            retrieval_role="answer_ready",
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::definition",
            document_id=document_id,
            chunk_index=0,
            chunk_kind="glossary_definition",
            title=f"{entry['term']} definition",
            text=text,
            source_paths=entry["related_sources"],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[],
            phase_tags=[],
            page_routes=[],
            authority_level="terminology",
            retrieval_role="answer_ready",
        )


def _build_source_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    sources = load_all_navigator_registries().source_registry["sources"]
    for source in sources:
        document_id = f"source::{source['source_id']}"
        page_routes = _page_routes_from_names(source["consumer_pages"])
        runtime_scope = source["runtime_scope"]
        runtime_scope_lines = runtime_scope if isinstance(runtime_scope, list) else [runtime_scope]
        caveats = []
        retrieval_role = "answer_ready"
        exclusion_matches = _get_exclusion_matches(source["path"])
        special_handling_matches = _get_special_handling_matches(source["path"])
        if str(source["source_kind"]).startswith("external_legacy_reference"):
            retrieval_role = "reference_only"
            caveats.append("External legacy reference only; content is not locally ingested.")
        elif exclusion_matches:
            retrieval_role = "reference_only"
            caveats.extend(match["reason"] for match in exclusion_matches)
        caveats.extend(match["handling"] for match in special_handling_matches)

        text = (
            f"Source kind: {source['source_kind']}\n"
            f"Repo layer: {source['repo_layer']}\n"
            f"Authority level: {source['authority_level']}\n\n"
            f"Canonical scope:\n- " + "\n- ".join(source["canonical_scope"])
            + "\n\nRuntime scope:\n- "
            + "\n- ".join(runtime_scope_lines)
            + "\n\nNotes: "
            + source["notes"]
        )
        _append_document(
            documents,
            document_id=document_id,
            document_class="source_entry",
            title=source["title"],
            content=text,
            source_paths=[source["path"]],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[],
            phase_tags=[str(phase).lower() for phase in source["phase_links"]],
            page_routes=page_routes,
            authority_level=source["authority_level"],
            retrieval_role=retrieval_role,
            caveats=caveats,
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::overview",
            document_id=document_id,
            chunk_index=0,
            chunk_kind="source_overview",
            title=f"{source['title']} overview",
            text=text,
            source_paths=[source["path"]],
            registry_refs=[document_id],
            truth_tags=[],
            drift_tags=[],
            phase_tags=[str(phase).lower() for phase in source["phase_links"]],
            page_routes=page_routes,
            authority_level=source["authority_level"],
            retrieval_role=retrieval_role,
            caveats=caveats,
        )


def _build_markdown_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    repo_root = get_repo_root()
    for relative_path, retrieval_role in _MARKDOWN_SOURCES:
        path = repo_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        sections = _iter_markdown_sections(text)
        caveats = []
        if relative_path == "navigator/repo_rename_assessment.md":
            caveats.append("Recommendation-only governance note; not runtime or model truth.")
        for section_index, (section_title, section_body) in enumerate(sections):
            document_id = f"markdown::{_slugify(relative_path)}::{_slugify(section_title)}"
            _append_document(
                documents,
                document_id=document_id,
                document_class="markdown_section",
                title=f"{relative_path} :: {section_title}",
                content=section_body,
                source_paths=[relative_path],
                registry_refs=[],
                truth_tags=[],
                drift_tags=[],
                phase_tags=[],
                page_routes=[],
                authority_level="documentation",
                retrieval_role=retrieval_role,
                caveats=caveats,
            )
            _append_chunk(
                chunks,
                chunk_id=f"{document_id}::body",
                document_id=document_id,
                chunk_index=section_index,
                chunk_kind="markdown_section",
                title=f"{section_title} body",
                text=section_body,
                source_paths=[relative_path],
                registry_refs=[],
                truth_tags=[],
                drift_tags=[],
                phase_tags=[],
                page_routes=[],
                authority_level="documentation",
                retrieval_role=retrieval_role,
                caveats=caveats,
            )


def _summarize_contract_json(payload: dict[str, Any]) -> str:
    required_artifacts = sorted(payload.get("required_artifacts", {}).keys())
    optional_artifacts = sorted(payload.get("optional_artifacts", {}).keys())
    runtime_policy = payload.get("runtime_policy", {})
    join_strategy = payload.get("join_strategy", {})
    return (
        "Runtime policy:\n- "
        + "\n- ".join(f"{key}: {value}" for key, value in runtime_policy.items())
        + "\n\nJoin strategy:\n"
        + f"Primary row key: {join_strategy.get('primary_row_key')}\n"
        + f"Strategy: {join_strategy.get('strategy')}\n\n"
        + "Required artifacts:\n- "
        + "\n- ".join(required_artifacts)
        + "\n\nOptional artifacts:\n- "
        + "\n- ".join(optional_artifacts)
    )


def _summarize_metadata_json(payload: dict[str, Any]) -> str:
    lines = [
        f"Project name: {payload.get('project_name')}",
        f"Final model: {payload.get('final_model')}",
        f"Selected threshold: {payload.get('selected_threshold')}",
        f"Model mode: {payload.get('model_mode_main')}",
        f"Artifact version: {payload.get('artifact_version')}",
    ]
    notes = payload.get("notes", [])
    if notes:
        lines.append("Notes:\n- " + "\n- ".join(notes))
    return "\n".join(lines)


def _summarize_model_modes_json(payload: dict[str, Any]) -> str:
    operational = payload.get("operational", {})
    survey_rich = payload.get("survey_rich", {})
    lines = [
        "Operational mode:",
        f"- Feature inclusion: {operational.get('feature_inclusion_summary', 'n/a')}",
        f"- Deployment notes: {operational.get('deployment_notes', 'n/a')}",
        "",
        "Survey-rich mode:",
        f"- Feature inclusion: {survey_rich.get('feature_inclusion_summary', 'n/a')}",
        f"- Deployment notes: {survey_rich.get('deployment_notes', 'n/a')}",
        "",
        f"Comparison notes: {payload.get('comparison_notes', 'n/a')}",
        f"Source workflow: {payload.get('source_workflow', 'n/a')}",
    ]
    return "\n".join(lines)


def _build_small_json_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    repo_root = get_repo_root()
    for relative_path, retrieval_role in _SMALL_JSON_SOURCES:
        path = repo_root / relative_path
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        document_id = f"json::{_slugify(relative_path)}"
        caveats = []
        if relative_path in {
            "artifacts/v2/metadata.json",
            "artifacts/v2/model_modes_summary.json",
        }:
            caveats.append("Contains machine-specific provenance text that should be treated carefully.")

        if relative_path.endswith("artifact_contract.json"):
            summary_text = _summarize_contract_json(payload)
            truth_tags = ["runtime_truth"]
            drift_tags = ["drift_contract_optional_model_modes_schema"]
        elif relative_path.endswith("metadata.json"):
            summary_text = _summarize_metadata_json(payload)
            truth_tags = ["public_model_truth", "runtime_truth"]
            drift_tags = ["drift_public_selection_vs_rerun_leader"]
        else:
            summary_text = _summarize_model_modes_json(payload)
            truth_tags = []
            drift_tags = ["drift_contract_optional_model_modes_schema"]

        _append_document(
            documents,
            document_id=document_id,
            document_class="artifact_metadata",
            title=relative_path,
            content=summary_text,
            source_paths=[relative_path],
            registry_refs=[],
            truth_tags=truth_tags,
            drift_tags=drift_tags,
            phase_tags=[],
            page_routes=[],
            authority_level="artifact_metadata",
            retrieval_role=retrieval_role,
            caveats=caveats,
        )
        _append_chunk(
            chunks,
            chunk_id=f"{document_id}::overview",
            document_id=document_id,
            chunk_index=0,
            chunk_kind="artifact_metadata",
            title=f"{relative_path} overview",
            text=summary_text,
            source_paths=[relative_path],
            registry_refs=[],
            truth_tags=truth_tags,
            drift_tags=drift_tags,
            phase_tags=[],
            page_routes=[],
            authority_level="artifact_metadata",
            retrieval_role=retrieval_role,
            caveats=caveats,
        )


def build_retrieval_pack() -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    _build_truth_documents(documents, chunks)
    _build_drift_documents(documents, chunks)
    _build_phase_documents(documents, chunks)
    _build_glossary_documents(documents, chunks)
    _build_source_documents(documents, chunks)
    _build_markdown_documents(documents, chunks)
    _build_small_json_documents(documents, chunks)

    documents.sort(key=lambda item: item["document_id"])
    chunks.sort(key=lambda item: item["chunk_id"])

    eligibility_policy = {
        "policy_version": "1.0",
        "included_content": [
            {
                "category": "registry_entities",
                "description": "Truth entries, drift items, phase entries, glossary terms, and source inventory entries are eligible because they are already governed and structured.",
            },
            {
                "category": "curated_markdown_sections",
                "description": "README, artifact guide, navigator docs, and navigator notes are eligible as section-level documentation chunks.",
            },
            {
                "category": "small_json_metadata",
                "description": "Small metadata and contract JSON files are eligible as summarized retrieval documents with caveats where needed.",
            },
        ],
        "excluded_content": _EXCLUDED_CONTENT,
        "special_handling": _SPECIAL_HANDLING,
    }

    manifest = {
        "pack_version": "1.0",
        "generator_module": "app.services.navigator_retrieval_pack",
        "generator_script": "scripts/build_retrieval_pack.py",
        "output_root": "navigator/retrieval_pack",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "document_class_counts": dict(sorted(Counter(item["document_class"] for item in documents).items())),
        "chunk_kind_counts": dict(sorted(Counter(item["chunk_kind"] for item in chunks).items())),
        "stable_id_strategy": "Deterministic ids derived from registry ids, source ids, normalized paths, and structural section names.",
        "chunking_policy": {
            "registry_entries": "Chunk boundaries follow individual truth, drift, phase, glossary, and source entry boundaries.",
            "markdown_documents": "Chunk boundaries follow markdown heading sections instead of raw token windows.",
            "artifact_metadata": "Chunk boundaries follow small metadata file summaries rather than raw table dumps.",
            "length_guard": "WP5 keeps chunks structure-aware and compact; no embedding-oriented token chunking is performed.",
        },
        "eligibility_policy_file": "navigator/retrieval_pack/eligibility_policy.json",
        "output_files": {
            "manifest": "navigator/retrieval_pack/manifest.json",
            "eligibility_policy": "navigator/retrieval_pack/eligibility_policy.json",
            "documents": "navigator/retrieval_pack/documents.jsonl",
            "chunks": "navigator/retrieval_pack/chunks.jsonl",
        },
    }

    return {
        "manifest": manifest,
        "eligibility_policy": eligibility_policy,
        "documents": documents,
        "chunks": chunks,
    }


def write_retrieval_pack(output_root: Path | None = None) -> dict[str, Any]:
    pack = build_retrieval_pack()
    root = output_root or get_retrieval_pack_root()
    root.mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps(pack["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "eligibility_policy.json").write_text(
        json.dumps(pack["eligibility_policy"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (root / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for item in pack["documents"]:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    with (root / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for item in pack["chunks"]:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    return {
        "output_root": root,
        "document_count": len(pack["documents"]),
        "chunk_count": len(pack["chunks"]),
    }


def load_retrieval_pack(output_root: Path | None = None) -> dict[str, Any]:
    root = output_root or get_retrieval_pack_root()
    manifest_path = root / "manifest.json"
    eligibility_path = root / "eligibility_policy.json"
    documents_path = root / "documents.jsonl"
    chunks_path = root / "chunks.jsonl"

    return {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "eligibility_policy": json.loads(eligibility_path.read_text(encoding="utf-8")),
        "documents": [
            json.loads(line)
            for line in documents_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "chunks": [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    }
