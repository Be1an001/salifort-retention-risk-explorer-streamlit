from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_system_readiness import (
    build_demo_readiness_checklist,
    build_execution_eligibility_summary,
    build_system_readiness_report,
    load_system_readiness_registries,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        registries = load_system_readiness_registries()
        readiness = build_system_readiness_report()
        checklist = build_demo_readiness_checklist()
        execution = build_execution_eligibility_summary()

        status_vocab = set(registries["approval_gates"]["status_vocabulary"])
        _assert(
            {"ready", "review_needed", "blocked", "preview_only"}.issubset(status_vocab),
            "Shared status vocabulary is incomplete.",
        )
        _assert(
            readiness["summary"]["component_count"]
            == len(registries["system_readiness"]["readiness_components"]),
            "Readiness component count mismatch.",
        )
        _assert(
            all(row["execution_allowed_in_streamlit"] is False for row in readiness["component_rows"]),
            "A readiness component allowed Streamlit execution.",
        )
        _assert(
            execution["summary"]["streamlit_executable_workflows"] == 0,
            "At least one workflow is executable from Streamlit.",
        )
        _assert(
            checklist["status"] in {"ready", "blocked"},
            "Demo checklist returned unexpected status.",
        )
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"System readiness validation failed: {type(exc).__name__}: {exc}")
        return 1

    print("System readiness validation passed.")
    print(f"- Components: {readiness['summary']['component_count']}")
    print(f"- Approval gates: {readiness['summary']['approval_gate_count']}")
    print(f"- Demo status: {readiness['summary']['demo_status']}")
    print(f"- Status counts: {readiness['status_counts']}")
    print(f"- Streamlit executable workflows: {execution['summary']['streamlit_executable_workflows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
