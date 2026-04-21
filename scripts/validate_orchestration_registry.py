from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_orchestration import (
    build_runbook_view,
    list_workflows,
    load_orchestration_registries,
)

_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|api[_-]?key\s*[:=]\s*['\"][^'\"]{12,})",
    re.IGNORECASE,
)
_PROHIBITED_TERMS = ("agent_loop", "free_text_chat", "unrestricted_browser")


def main() -> int:
    try:
        registries = load_orchestration_registries()
        task_registry_text = (
            REPO_ROOT / "navigator" / "orchestration" / "task_registry.json"
        ).read_text(encoding="utf-8")
        workflow_registry_text = (
            REPO_ROOT / "navigator" / "orchestration" / "workflow_registry.json"
        ).read_text(encoding="utf-8")
        registry_text = task_registry_text + "\n" + workflow_registry_text
        if _SECRET_PATTERN.search(registry_text):
            raise ValueError("Secret-like value found in orchestration registries.")
        lowered = registry_text.lower()
        prohibited = [term for term in _PROHIBITED_TERMS if term in lowered]
        if prohibited:
            raise ValueError(f"Prohibited orchestration term(s) found: {prohibited}")

        workflows = list_workflows()
        runbooks = [build_runbook_view(workflow["workflow_id"]) for workflow in workflows]
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"Orchestration registry validation failed: {type(exc).__name__}: {exc}")
        return 1

    print("Orchestration registry validation passed.")
    print(f"- Tasks: {len(registries['task_registry']['tasks'])}")
    print(f"- Workflows: {len(workflows)}")
    for runbook in runbooks:
        summary = runbook["summary"]
        blockers = runbook["blockers"]
        print(
            f"- {summary['workflow_id']}: tasks={summary['task_count']} "
            f"runtime={summary['runtime_mode']} scheduler={summary['scheduler_eligibility']} "
            f"blocker_status={blockers['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
