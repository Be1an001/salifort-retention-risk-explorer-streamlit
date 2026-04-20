from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_answer_assembly import assemble_governed_answer
from app.services.navigator_embedding_index import (
    NavigatorEmbeddingConfigurationError,
    NavigatorEmbeddingIndexError,
    NavigatorEmbeddingRequestError,
    OpenAIEmbeddingConfig,
)
from app.services.navigator_retriever import get_retrieval_evaluation_queries


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    try:
        config = OpenAIEmbeddingConfig.from_env()
    except NavigatorEmbeddingConfigurationError as exc:
        print(f"BLOCKED: {exc}")
        return 2

    scenario_results = []
    for spec in get_retrieval_evaluation_queries():
        try:
            first = assemble_governed_answer(spec.query, config=config)
            second = assemble_governed_answer(spec.query, config=config)
        except NavigatorEmbeddingRequestError as exc:
            print(f"BLOCKED: {exc}")
            return 2
        except NavigatorEmbeddingIndexError as exc:
            print(f"FAILED: {exc}")
            return 1

        first_payload = first.as_dict()
        second_payload = second.as_dict()
        _require(
            first_payload == second_payload,
            f"Answer assembly is not deterministic for query {spec.query!r}.",
        )
        _require(first.direct_answer.strip(), f"Direct answer is empty for {spec.query!r}.")
        _require(first.citations, f"No citations produced for {spec.query!r}.")
        _require(first.source_paths, f"No source paths produced for {spec.query!r}.")

        matched_groups = set(first.coverage_summary["matched_groups"])
        missing_groups = set(first.coverage_summary["missing_groups"])
        expected_groups = {group.description for group in spec.expected_groups}
        _require(
            matched_groups.issubset(expected_groups),
            f"Unexpected matched groups for {spec.query!r}: {sorted(matched_groups - expected_groups)}",
        )
        _require(
            missing_groups.issubset(expected_groups),
            f"Unexpected missing groups for {spec.query!r}: {sorted(missing_groups - expected_groups)}",
        )

        if spec.query == "how is fallback different from final model truth":
            _require(
                "fallback_separation_preserved" in first.governance_flags,
                "Fallback-vs-final answer did not preserve the fallback separation governance flag.",
            )
            _require(
                first.coverage_summary["matched_groups"]
                == ["Fallback-truth chunk", "Fallback-vs-runtime drift chunk"],
                "Fallback-vs-final answer did not preserve both truth and drift dimensions.",
            )
            _require(
                any(citation.drift_tags for citation in first.citations),
                "Fallback-vs-final answer is missing a drift citation.",
            )

        if spec.query == "which page should be inspected for department exposure":
            page_routes = [page.route for page in first.recommended_pages]
            _require(
                "manager-action-view" in page_routes,
                "Department exposure answer did not recommend Manager Action View.",
            )

        if spec.query == "what is the public model truth":
            _require(
                any("public_model_truth" in citation.truth_tags for citation in first.citations),
                "Public model truth answer is missing a canonical truth citation.",
            )

        if spec.query == "why is threshold 0.29 used":
            _require(
                "0.29" in first.direct_answer,
                "Threshold answer did not explicitly preserve threshold 0.29.",
            )
            _require(
                "weighted XGBoost" in first.direct_answer,
                "Threshold answer did not preserve weighted XGBoost as the public model context.",
            )

        scenario_results.append(
            {
                "query": spec.query,
                "assembly_status": first.assembly_status,
                "matched_groups": first.coverage_summary["matched_groups"],
                "top_citation": first.citations[0].chunk_id,
            }
        )

    print("Governed answer assembly validation passed.")
    for item in scenario_results:
        print(
            f"- {item['query']}: status={item['assembly_status']} "
            f"groups={item['matched_groups']} top_citation={item['top_citation']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
