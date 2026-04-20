from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.navigator_embedding_index import (
    NavigatorEmbeddingIndexError,
    OpenAIEmbeddingConfig,
    _embed_text_batch,
    _normalize_vectors,
    load_retrieval_index,
)


@dataclass(frozen=True)
class RetrievalQuerySpec:
    query: str
    description: str
    expected_truth_tags: tuple[str, ...] = ()
    expected_phase_tags: tuple[str, ...] = ()
    expected_retrieval_role: str | None = None
    expected_page_routes: tuple[str, ...] = ()
    expected_drift_tags: tuple[str, ...] = ()


def get_retrieval_evaluation_queries() -> list[RetrievalQuerySpec]:
    return [
        RetrievalQuerySpec(
            query="what is the public model truth",
            description="Should retrieve governed public-model truth chunks.",
            expected_truth_tags=("public_model_truth",),
        ),
        RetrievalQuerySpec(
            query="why is threshold 0.29 used",
            description="Should retrieve chunks tied to the public threshold framing and metadata.",
            expected_truth_tags=("public_model_truth",),
        ),
        RetrievalQuerySpec(
            query="how is fallback different from final model truth",
            description="Should retrieve fallback/runtime separation chunks.",
            expected_truth_tags=("fallback_truth",),
            expected_drift_tags=("drift_runtime_rows_vs_fallback_rows",),
        ),
        RetrievalQuerySpec(
            query="which page should be inspected for department exposure",
            description="Should retrieve manager-action-view-oriented chunks.",
            expected_page_routes=("manager-action-view",),
        ),
        RetrievalQuerySpec(
            query="what drift exists between public selection and local rerun leader",
            description="Should retrieve the public-selection drift chunk.",
            expected_drift_tags=("drift_public_selection_vs_rerun_leader",),
        ),
    ]


def _matches_optional_filter(values: list[str], expected: str | None) -> bool:
    if expected is None:
        return True
    return expected in values


def _matches_path_filter(source_paths: list[str], expected: str | None) -> bool:
    if expected is None:
        return True
    normalized_expected = expected.lower()
    return any(normalized_expected in path.lower() for path in source_paths)


def retrieve_governed_chunks(
    query: str,
    *,
    config: OpenAIEmbeddingConfig,
    top_k: int = 5,
    truth_tag: str | None = None,
    phase_tag: str | None = None,
    retrieval_role: str | None = None,
    source_path_contains: str | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise NavigatorEmbeddingIndexError("Query text must not be empty.")

    index = load_retrieval_index()
    chunk_index = index["chunk_index"]
    corpus_vectors = index["vectors"]

    filtered_positions = [
        position
        for position, chunk in enumerate(chunk_index)
        if _matches_optional_filter(chunk["truth_tags"], truth_tag)
        and _matches_optional_filter(chunk["phase_tags"], phase_tag)
        and (retrieval_role is None or chunk["retrieval_role"] == retrieval_role)
        and _matches_path_filter(chunk["source_paths"], source_path_contains)
    ]
    if not filtered_positions:
        return []

    query_vector, _ = _embed_text_batch([query], config=config)
    normalized_query = _normalize_vectors(query_vector)[0]

    scored_results = []
    for position in filtered_positions:
        chunk = chunk_index[position]
        score = float(corpus_vectors[position].dot(normalized_query))
        scored_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "similarity_score": score,
                "source_paths": chunk["source_paths"],
                "registry_refs": chunk["registry_refs"],
                "truth_tags": chunk["truth_tags"],
                "drift_tags": chunk["drift_tags"],
                "phase_tags": chunk["phase_tags"],
                "page_routes": chunk["page_routes"],
                "retrieval_role": chunk["retrieval_role"],
                "title": chunk["title"],
                "text_preview": chunk["text_preview"],
                "authority_level": chunk["authority_level"],
                "caveats": chunk["caveats"],
            }
        )

    scored_results.sort(
        key=lambda item: (-item["similarity_score"], item["chunk_id"])
    )
    return scored_results[:top_k]


def get_retrieval_pack_chunk_count() -> int:
    from app.services.navigator_retrieval_pack import load_retrieval_pack

    return len(load_retrieval_pack()["chunks"])
