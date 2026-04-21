from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_agent_shell import build_agent_plan_preview, get_controlled_agent_requests
from app.services.navigator_orchestration import list_workflows
from app.services.navigator_system_readiness import build_system_readiness_report

_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|api[_-]?key\s*[:=]\s*['\"][^'\"]{12,})",
    re.IGNORECASE,
)
_STREAMLIT_EXECUTION_IMPORTS = (
    "import subprocess",
    "from subprocess",
    "os.system",
    "Popen(",
    "BashOperator",
    "PythonOperator",
)
_GOVERNANCE_TERMS = (
    "preview only",
    "does not execute",
    "human review",
    "fallback",
    "threshold 0.29",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        registry_paths = [
            REPO_ROOT / "navigator" / "agent" / "agent_policy.json",
            REPO_ROOT / "navigator" / "agent" / "request_catalog.json",
            REPO_ROOT / "navigator" / "system" / "approval_gates.json",
            REPO_ROOT / "navigator" / "system" / "system_readiness.json",
            REPO_ROOT / "navigator" / "orchestration" / "task_registry.json",
            REPO_ROOT / "navigator" / "orchestration" / "workflow_registry.json",
        ]
        combined_text = "\n".join(path.read_text(encoding="utf-8") for path in registry_paths)
        _assert(not _SECRET_PATTERN.search(combined_text), "Secret-like value found in governance registries.")

        page_text = (REPO_ROOT / "app" / "pages" / "pace_navigator.py").read_text(encoding="utf-8")
        prohibited_imports = [
            pattern for pattern in _STREAMLIT_EXECUTION_IMPORTS if pattern in page_text
        ]
        _assert(
            not prohibited_imports,
            f"Streamlit page contains execution-capable import/call pattern(s): {prohibited_imports}",
        )
        lowered_page = page_text.lower()
        missing_terms = [term for term in _GOVERNANCE_TERMS if term not in lowered_page]
        _assert(
            not missing_terms,
            f"Streamlit page is missing final governance wording: {missing_terms}",
        )

        requests = get_controlled_agent_requests()
        for request in requests:
            preview = build_agent_plan_preview(request["request_id"])
            _assert(preview["execution_allowed"] is False, f"Agent preview allowed execution for {request['request_id']}.")
        readiness = build_system_readiness_report()
        _assert(readiness["summary"]["component_count"] >= 8, "Expected integrated readiness components.")
        workflows = list_workflows()
        _assert(workflows, "No orchestration workflows loaded.")
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"Final governance validation failed: {type(exc).__name__}: {exc}")
        return 1

    print("Final governance validation passed.")
    print(f"- Controlled agent requests: {len(requests)}")
    print(f"- Orchestration workflows: {len(workflows)}")
    print(f"- Readiness components: {readiness['summary']['component_count']}")
    print("- Streamlit page has no execution-capable imports/calls from the checked list.")
    print("- Governance registries contain no secret-like values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
