from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_embedding_index import (
    NavigatorEmbeddingConfigurationError,
    NavigatorEmbeddingIndexError,
    NavigatorEmbeddingRequestError,
    NavigatorRetrievalIndexNotFoundError,
    OpenAIEmbeddingConfig,
    load_retrieval_index,
)
from app.services.navigator_retrieval_pack import load_retrieval_pack
from app.services.navigator_retriever import (
    get_retrieval_evaluation_queries,
    retrieve_governed_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the governed local retrieval runtime."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to evaluate per query.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the evaluation harness and retrieval-pack linkage without live query embedding.",
    )
    return parser.parse_args()


def _require_fields(record: dict[str, object], fields: set[str], record_name: str) -> None:
    missing = fields - set(record)
    if missing:
        raise RuntimeError(f"{record_name} missing required fields: {sorted(missing)}")


def _result_matches_expectation(result: dict[str, object], query_spec) -> bool:
    if query_spec.expected_truth_tags and not set(query_spec.expected_truth_tags).intersection(
        result["truth_tags"]
    ):
        return False
    if query_spec.expected_phase_tags and not set(query_spec.expected_phase_tags).intersection(
        result["phase_tags"]
    ):
        return False
    if query_spec.expected_page_routes and not set(query_spec.expected_page_routes).intersection(
        result["page_routes"]
    ):
        return False
    if query_spec.expected_drift_tags and not set(query_spec.expected_drift_tags).intersection(
        result["drift_tags"]
    ):
        return False
    if query_spec.expected_retrieval_role and result["retrieval_role"] != query_spec.expected_retrieval_role:
        return False
    return True


def main() -> int:
    args = parse_args()

    retrieval_pack = load_retrieval_pack()
    retrieval_pack_chunk_count = len(retrieval_pack["chunks"])
    query_specs = get_retrieval_evaluation_queries()

    if args.dry_run:
        print("Dry run: retrieval runtime validation inputs are available.")
        print(f"- Retrieval pack chunks: {retrieval_pack_chunk_count}")
        print(f"- Evaluation queries: {len(query_specs)}")
        return 0

    try:
        index = load_retrieval_index()
    except NavigatorRetrievalIndexNotFoundError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except NavigatorEmbeddingIndexError as exc:
        print(f"FAILED: {exc}")
        return 1

    manifest = index["manifest"]
    chunk_index = index["chunk_index"]
    vectors = index["vectors"]

    if manifest["index_vector_count"] != len(chunk_index):
        raise RuntimeError("Index manifest vector count does not match chunk index length.")
    if vectors.shape[0] != len(chunk_index):
        raise RuntimeError("Vector row count does not match chunk index length.")
    if retrieval_pack_chunk_count != len(chunk_index):
        raise RuntimeError("Index chunk count does not match retrieval pack chunk count.")

    required_chunk_fields = {
        "chunk_id",
        "document_id",
        "chunk_kind",
        "title",
        "text_preview",
        "source_paths",
        "registry_refs",
        "truth_tags",
        "drift_tags",
        "phase_tags",
        "page_routes",
        "authority_level",
        "retrieval_role",
        "caveats",
    }
    chunk_ids: set[str] = set()
    for index_position, chunk in enumerate(chunk_index, start=1):
        _require_fields(chunk, required_chunk_fields, f"Index chunk #{index_position}")
        chunk_ids.add(chunk["chunk_id"])
        if not chunk["source_paths"] and not chunk["registry_refs"]:
            raise RuntimeError(
                f"Index chunk #{index_position} has no traceability metadata."
            )

    pack_chunk_ids = {chunk["chunk_id"] for chunk in retrieval_pack["chunks"]}
    if chunk_ids != pack_chunk_ids:
        raise RuntimeError("Index chunk ids do not align with retrieval pack chunk ids.")

    try:
        config = OpenAIEmbeddingConfig.from_env(
            embedding_model=manifest["embedding_model"],
            dimensions=manifest["embedding_dimensions_requested"],
        )
    except NavigatorEmbeddingConfigurationError as exc:
        print(f"BLOCKED: {exc}")
        return 2

    evaluation_results = []
    for query_spec in query_specs:
        try:
            results = retrieve_governed_chunks(
                query_spec.query,
                config=config,
                top_k=args.top_k,
            )
        except NavigatorEmbeddingRequestError as exc:
            print(f"BLOCKED: {exc}")
            return 2
        except NavigatorEmbeddingIndexError as exc:
            print(f"FAILED: {exc}")
            return 1

        matched = any(_result_matches_expectation(result, query_spec) for result in results)
        evaluation_results.append(
            {
                "query": query_spec.query,
                "matched_expectation": matched,
                "top_chunk_ids": [result["chunk_id"] for result in results],
            }
        )
        if not matched:
            raise RuntimeError(
                f"Query {query_spec.query!r} did not return an obviously relevant governed chunk in top-{args.top_k}."
            )

    print("Retrieval runtime validation passed.")
    print(f"- Retrieval pack chunks: {retrieval_pack_chunk_count}")
    print(f"- Indexed chunks: {len(chunk_index)}")
    print(f"- Vector rows: {vectors.shape[0]}")
    print(f"- Embedding model: {manifest['embedding_model']}")
    print("- Query evaluation summary:")
    for item in evaluation_results:
        print(
            f"  - {item['query']}: matched={item['matched_expectation']} top_chunks={item['top_chunk_ids'][:3]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
