from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_agent_shell import (
    build_agent_plan_preview,
    get_agent_registry_paths,
    get_controlled_agent_requests,
    get_disallowed_agent_behaviors,
    get_supported_agent_intents,
    load_agent_shell_registries,
)
from app.services.navigator_orchestration import get_task_definition, get_workflow_definition

_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|api[_-]?key\s*[:=]\s*['\"][^'\"]{12,})",
    re.IGNORECASE,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        registries = load_agent_shell_registries()
        paths = get_agent_registry_paths()
        registry_text = "\n".join(
            path.read_text(encoding="utf-8") for path in paths.values()
        )
        _assert(
            not _SECRET_PATTERN.search(registry_text),
            "Secret-like value found in agent shell registries.",
        )

        intents = get_supported_agent_intents()
        requests = get_controlled_agent_requests()
        disallowed = get_disallowed_agent_behaviors()
        _assert(len(intents) >= 8, "Expected at least eight supported intents.")
        _assert(len(requests) >= 8, "Expected at least eight controlled requests.")
        _assert(disallowed, "Disallowed behaviors must be present.")

        intent_ids = {intent["intent_id"] for intent in intents}
        for request in requests:
            _assert(
                request["intent_id"] in intent_ids,
                f"Request {request['request_id']} has unknown intent.",
            )
            get_workflow_definition(request["workflow_id"])
            for task_id in request["task_ids"]:
                get_task_definition(task_id)
            preview = build_agent_plan_preview(request["request_id"])
            _assert(
                preview["status"] == "preview_ready",
                f"Plan preview did not build for {request['request_id']}.",
            )
            _assert(
                preview["execution_allowed"] is False,
                f"Execution was allowed for {request['request_id']}.",
            )
            _assert(
                preview["plan_steps"],
                f"No plan steps returned for {request['request_id']}.",
            )
            _assert(
                preview["guardrails"]["execution_policy"]["mode"] == "preview_only",
                f"Request {request['request_id']} is not preview-only.",
            )

        for unsupported in registries["request_catalog"]["unsupported_request_examples"]:
            preview = build_agent_plan_preview(unsupported["request_text"])
            _assert(
                preview["status"] == "blocked",
                f"Unsupported request was not blocked: {unsupported['request_text']}",
            )
            _assert(
                preview["execution_allowed"] is False,
                "Blocked unsupported request still allowed execution.",
            )
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"Agent shell validation failed: {type(exc).__name__}: {exc}")
        return 1

    print("Agent shell validation passed.")
    print(f"- Supported intents: {len(intents)}")
    print(f"- Controlled requests: {len(requests)}")
    print(f"- Disallowed behaviors: {len(disallowed)}")
    print("- All controlled requests returned preview-only plans.")
    print("- Unsupported/disallowed examples returned blocked previews.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
