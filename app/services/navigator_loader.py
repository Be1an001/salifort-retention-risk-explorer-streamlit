from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.navigator_types import (
    NavigatorRegistryBundle,
    NavigatorRegistryNotFoundError,
    NavigatorRegistryValidationError,
)

_TOP_LEVEL_SOURCE_KEYS = {"registry_version", "repo_name", "scope", "sources"}
_REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "title",
    "source_kind",
    "path",
    "repo_layer",
    "authority_level",
    "canonical_scope",
    "runtime_scope",
    "phase_links",
    "consumer_pages",
    "freshness_mode",
    "notes",
}
_PHASE_NAMES = {"plan", "analyze", "construct", "execute"}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_navigator_registry_paths() -> dict[str, Path]:
    navigator_root = get_repo_root() / "navigator"
    return {
        "source_registry": navigator_root / "source_registry.yaml",
        "truth_registry": navigator_root / "truth_registry.json",
        "drift_register": navigator_root / "drift_register.json",
        "pace_phase_map": navigator_root / "pace_phase_map.json",
        "glossary": navigator_root / "glossary.json",
    }


def clear_navigator_registry_caches() -> None:
    load_source_registry.cache_clear()
    load_truth_registry.cache_clear()
    load_drift_register.cache_clear()
    load_pace_phase_map.cache_clear()
    load_glossary.cache_clear()
    load_all_navigator_registries.cache_clear()


def _require_registry_path(name: str) -> Path:
    path = get_navigator_registry_paths()[name]
    if not path.exists():
        raise NavigatorRegistryNotFoundError(
            f"Required navigator registry '{name}' is missing at {path}."
        )
    return path


def _load_json_file(name: str) -> dict[str, Any]:
    path = _require_registry_path(name)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise NavigatorRegistryValidationError(
            f"Navigator registry '{name}' is not valid JSON: {exc}"
        ) from exc


def _parse_yaml_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped == "[]":
        return []
    if not stripped:
        return ""
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise NavigatorRegistryValidationError(
                f"Unable to parse YAML quoted scalar value: {stripped!r}"
            ) from exc
    return stripped


def _parse_source_registry_yaml(text: str) -> dict[str, Any]:
    if not text.strip():
        raise NavigatorRegistryValidationError("source_registry.yaml is empty.")

    lines = text.splitlines()
    parsed: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    current_source: dict[str, Any] | None = None
    current_list_key: str | None = None
    inside_sources = False

    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        if raw.startswith("\t"):
            raise NavigatorRegistryValidationError(
                f"Tabs are not allowed in source_registry.yaml (line {lineno})."
            )

        indent = len(raw) - len(raw.lstrip(" "))
        text_value = raw.strip()

        if indent == 0:
            current_list_key = None
            current_source = None

            if text_value == "sources:":
                inside_sources = True
                parsed["sources"] = sources
                continue

            if text_value.endswith(":"):
                raise NavigatorRegistryValidationError(
                    f"Unexpected top-level nested block at line {lineno}: {text_value}"
                )
            if ": " not in text_value:
                raise NavigatorRegistryValidationError(
                    f"Invalid top-level mapping entry at line {lineno}: {text_value}"
                )

            key, raw_value = text_value.split(": ", 1)
            parsed[key] = _parse_yaml_scalar(raw_value)
            continue

        if not inside_sources:
            raise NavigatorRegistryValidationError(
                f"Indented content before 'sources:' at line {lineno}."
            )

        if indent == 2 and text_value.startswith("- "):
            payload = text_value[2:]
            if ": " not in payload:
                raise NavigatorRegistryValidationError(
                    f"Invalid source entry start at line {lineno}: {text_value}"
                )
            key, raw_value = payload.split(": ", 1)
            current_source = {key: _parse_yaml_scalar(raw_value)}
            sources.append(current_source)
            current_list_key = None
            continue

        if current_source is None:
            raise NavigatorRegistryValidationError(
                f"Source field encountered before source item start at line {lineno}."
            )

        if indent == 4:
            if text_value.endswith(":"):
                current_list_key = text_value[:-1]
                current_source[current_list_key] = []
                continue

            if ": " not in text_value:
                raise NavigatorRegistryValidationError(
                    f"Invalid source field at line {lineno}: {text_value}"
                )

            key, raw_value = text_value.split(": ", 1)
            current_source[key] = _parse_yaml_scalar(raw_value)
            current_list_key = None
            continue

        if indent == 6 and text_value.startswith("- "):
            if current_list_key is None:
                raise NavigatorRegistryValidationError(
                    f"List item without active list field at line {lineno}."
                )
            list_value = current_source.get(current_list_key)
            if not isinstance(list_value, list):
                raise NavigatorRegistryValidationError(
                    f"Field '{current_list_key}' is not a list at line {lineno}."
                )
            list_value.append(_parse_yaml_scalar(text_value[2:]))
            continue

        raise NavigatorRegistryValidationError(
            f"Unsupported YAML indentation or structure at line {lineno}: {raw!r}"
        )

    missing_top_keys = _TOP_LEVEL_SOURCE_KEYS - set(parsed)
    if missing_top_keys:
        raise NavigatorRegistryValidationError(
            f"source_registry.yaml is missing top-level keys: {sorted(missing_top_keys)}"
        )

    if not isinstance(parsed["sources"], list) or not parsed["sources"]:
        raise NavigatorRegistryValidationError(
            "source_registry.yaml must contain a non-empty 'sources' list."
        )

    for index, source in enumerate(parsed["sources"], start=1):
        if not isinstance(source, dict):
            raise NavigatorRegistryValidationError(
                f"Source entry #{index} is not a mapping."
            )
        missing_fields = _REQUIRED_SOURCE_FIELDS - set(source)
        if missing_fields:
            raise NavigatorRegistryValidationError(
                f"Source entry #{index} is missing fields: {sorted(missing_fields)}"
            )
        for list_field in ("canonical_scope", "phase_links", "consumer_pages"):
            value = source[list_field]
            if not isinstance(value, list):
                raise NavigatorRegistryValidationError(
                    f"Source entry #{index} field '{list_field}' must be a list."
                )
        for phase_name in source["phase_links"]:
            if str(phase_name).strip().lower() not in _PHASE_NAMES:
                raise NavigatorRegistryValidationError(
                    f"Source entry #{index} contains unsupported phase link: {phase_name!r}"
                )

    return parsed


def _validate_truth_registry(payload: dict[str, Any]) -> dict[str, Any]:
    required_top_keys = {"registry_version", "repo_name", "truth_domains"}
    missing_keys = required_top_keys - set(payload)
    if missing_keys:
        raise NavigatorRegistryValidationError(
            f"truth_registry.json is missing keys: {sorted(missing_keys)}"
        )

    truth_domains = payload["truth_domains"]
    if not isinstance(truth_domains, dict) or not truth_domains:
        raise NavigatorRegistryValidationError(
            "truth_registry.json must contain a non-empty 'truth_domains' mapping."
        )

    required_fields = {
        "truth_id",
        "title",
        "description",
        "primary_sources",
        "secondary_sources",
        "authority_rule",
        "conflicts_with",
        "preserve_in_upgrade",
        "notes",
    }
    for domain_name, entry in truth_domains.items():
        if not isinstance(entry, dict):
            raise NavigatorRegistryValidationError(
                f"Truth domain '{domain_name}' must map to an object."
            )
        missing_fields = required_fields - set(entry)
        if missing_fields:
            raise NavigatorRegistryValidationError(
                f"Truth domain '{domain_name}' is missing fields: {sorted(missing_fields)}"
            )
    return payload


def _validate_drift_register(payload: dict[str, Any]) -> dict[str, Any]:
    required_top_keys = {"registry_version", "repo_name", "drifts"}
    missing_keys = required_top_keys - set(payload)
    if missing_keys:
        raise NavigatorRegistryValidationError(
            f"drift_register.json is missing keys: {sorted(missing_keys)}"
        )
    drifts = payload["drifts"]
    if not isinstance(drifts, list) or not drifts:
        raise NavigatorRegistryValidationError(
            "drift_register.json must contain a non-empty 'drifts' list."
        )

    required_fields = {
        "drift_id",
        "title",
        "severity",
        "status",
        "canonical_side",
        "current_side",
        "source_evidence",
        "user_visible_risk",
        "upgrade_handling_rule",
        "notes",
    }
    for index, item in enumerate(drifts, start=1):
        if not isinstance(item, dict):
            raise NavigatorRegistryValidationError(
                f"Drift item #{index} must be an object."
            )
        missing_fields = required_fields - set(item)
        if missing_fields:
            raise NavigatorRegistryValidationError(
                f"Drift item #{index} is missing fields: {sorted(missing_fields)}"
            )
    return payload


def _validate_pace_phase_map(payload: dict[str, Any]) -> dict[str, Any]:
    required_top_keys = {"map_version", "project_name", "phase_order", "phases"}
    missing_keys = required_top_keys - set(payload)
    if missing_keys:
        raise NavigatorRegistryValidationError(
            f"pace_phase_map.json is missing keys: {sorted(missing_keys)}"
        )

    phase_order = payload["phase_order"]
    phases = payload["phases"]
    if not isinstance(phase_order, list) or not isinstance(phases, list) or not phases:
        raise NavigatorRegistryValidationError(
            "pace_phase_map.json must contain non-empty 'phase_order' and 'phases' lists."
        )

    required_fields = {
        "phase_id",
        "phase_title",
        "phase_goal",
        "portfolio_meaning",
        "old_repo_method_links",
        "new_repo_runtime_links",
        "app_pages",
        "artifacts",
        "figures",
        "key_takeaways",
        "known_drifts",
        "future_navigator_use",
    }
    seen_phase_ids: set[str] = set()
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            raise NavigatorRegistryValidationError(
                f"PACE phase #{index} must be an object."
            )
        missing_fields = required_fields - set(phase)
        if missing_fields:
            raise NavigatorRegistryValidationError(
                f"PACE phase #{index} is missing fields: {sorted(missing_fields)}"
            )
        phase_id = str(phase["phase_id"]).strip().lower()
        seen_phase_ids.add(phase_id)

    missing_phase_ids = _PHASE_NAMES - seen_phase_ids
    if missing_phase_ids:
        raise NavigatorRegistryValidationError(
            f"pace_phase_map.json is missing required PACE phases: {sorted(missing_phase_ids)}"
        )
    return payload


def _validate_glossary(payload: dict[str, Any]) -> dict[str, Any]:
    required_top_keys = {"glossary_version", "terms"}
    missing_keys = required_top_keys - set(payload)
    if missing_keys:
        raise NavigatorRegistryValidationError(
            f"glossary.json is missing keys: {sorted(missing_keys)}"
        )
    terms = payload["terms"]
    if not isinstance(terms, list) or not terms:
        raise NavigatorRegistryValidationError(
            "glossary.json must contain a non-empty 'terms' list."
        )
    required_fields = {
        "term",
        "definition",
        "preferred_usage",
        "avoid_confusion_with",
        "related_sources",
    }
    for index, term in enumerate(terms, start=1):
        if not isinstance(term, dict):
            raise NavigatorRegistryValidationError(
                f"Glossary entry #{index} must be an object."
            )
        missing_fields = required_fields - set(term)
        if missing_fields:
            raise NavigatorRegistryValidationError(
                f"Glossary entry #{index} is missing fields: {sorted(missing_fields)}"
            )
    return payload


@lru_cache(maxsize=1)
def load_source_registry() -> dict[str, Any]:
    path = _require_registry_path("source_registry")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NavigatorRegistryValidationError(
            f"Unable to read source_registry.yaml: {exc}"
        ) from exc
    return _parse_source_registry_yaml(text)


@lru_cache(maxsize=1)
def load_truth_registry() -> dict[str, Any]:
    return _validate_truth_registry(_load_json_file("truth_registry"))


@lru_cache(maxsize=1)
def load_drift_register() -> dict[str, Any]:
    return _validate_drift_register(_load_json_file("drift_register"))


@lru_cache(maxsize=1)
def load_pace_phase_map() -> dict[str, Any]:
    return _validate_pace_phase_map(_load_json_file("pace_phase_map"))


@lru_cache(maxsize=1)
def load_glossary() -> dict[str, Any]:
    return _validate_glossary(_load_json_file("glossary"))


@lru_cache(maxsize=1)
def load_all_navigator_registries() -> NavigatorRegistryBundle:
    paths = get_navigator_registry_paths()
    return NavigatorRegistryBundle(
        source_registry=load_source_registry(),
        truth_registry=load_truth_registry(),
        drift_register=load_drift_register(),
        pace_phase_map=load_pace_phase_map(),
        glossary=load_glossary(),
        registry_paths=paths,
    )
