from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_loader import load_all_navigator_registries
from app.services.navigator_queries import (
    get_drift_items,
    get_pace_phase,
    get_runtime_governance_summary,
    get_truth_entries,
    lookup_glossary,
    recommend_page_for_topic,
)


def main() -> int:
    try:
        bundle = load_all_navigator_registries()
        truth_entries = get_truth_entries()
        high_drifts = get_drift_items(severity="high")
        construct_phase = get_pace_phase("construct")
        glossary_match = lookup_glossary("artifact backed mode")
        page_recommendation = recommend_page_for_topic("threshold tradeoffs and confusion matrix")
        governance = get_runtime_governance_summary()
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"Navigator validation failed: {type(exc).__name__}: {exc}")
        return 1

    print("Navigator layer validation passed.")
    print(f"- Loaded source records: {len(bundle.source_registry['sources'])}")
    print(f"- Loaded truth entries: {len(truth_entries)}")
    print(f"- Loaded drift items: {len(bundle.drift_register['drifts'])}")
    print(f"- High severity drifts: {len(high_drifts)}")
    print(f"- Construct phase pages: {', '.join(construct_phase['app_pages'])}")
    print(f"- Glossary lookup: {glossary_match['term'] if glossary_match else 'None'}")
    print(
        "- Topic recommendation: "
        f"{page_recommendation['recommended_page_title']} "
        f"({page_recommendation['recommended_page_route']})"
    )
    print(
        "- Governance summary truths: "
        f"{governance['public_model_truth']['truth_id']}, "
        f"{governance['artifact_backed_runtime_truth']['truth_id']}, "
        f"{governance['fallback_truth']['truth_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
