from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_retrieval_pack import build_retrieval_pack, load_retrieval_pack


def _require_fields(record: dict[str, object], fields: set[str], record_name: str) -> None:
    missing = fields - set(record)
    if missing:
        raise RuntimeError(f"{record_name} missing required fields: {sorted(missing)}")


def main() -> int:
    built = build_retrieval_pack()
    written = load_retrieval_pack()

    if built["manifest"] != written["manifest"]:
        raise RuntimeError("Written manifest does not match the current deterministic build output.")
    if built["eligibility_policy"] != written["eligibility_policy"]:
        raise RuntimeError("Written eligibility policy does not match the current deterministic build output.")
    if built["documents"] != written["documents"]:
        raise RuntimeError("Written documents.jsonl does not match the current deterministic build output.")
    if built["chunks"] != written["chunks"]:
        raise RuntimeError("Written chunks.jsonl does not match the current deterministic build output.")

    document_ids = [item["document_id"] for item in written["documents"]]
    chunk_ids = [item["chunk_id"] for item in written["chunks"]]
    if len(document_ids) != len(set(document_ids)):
        raise RuntimeError("Document ids are not unique.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Chunk ids are not unique.")

    required_document_fields = {
        "document_id",
        "document_class",
        "title",
        "content",
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
    required_chunk_fields = {
        "chunk_id",
        "document_id",
        "chunk_index",
        "chunk_kind",
        "title",
        "text",
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

    for index, document in enumerate(written["documents"], start=1):
        _require_fields(document, required_document_fields, f"Document #{index}")

    known_truth_tags = {
        "project_identity_truth",
        "public_model_truth",
        "method_truth",
        "runtime_truth",
        "fallback_truth",
        "orchestration_truth",
    }
    known_phase_tags = {"plan", "analyze", "construct", "execute"}
    for index, chunk in enumerate(written["chunks"], start=1):
        _require_fields(chunk, required_chunk_fields, f"Chunk #{index}")
        if chunk["document_id"] not in document_ids:
            raise RuntimeError(f"Chunk #{index} references unknown document id {chunk['document_id']!r}.")
        if not chunk["source_paths"] and not chunk["registry_refs"]:
            raise RuntimeError(f"Chunk #{index} is missing traceability to source paths or registry refs.")
        unknown_truth_tags = set(chunk["truth_tags"]) - known_truth_tags
        if unknown_truth_tags:
            raise RuntimeError(f"Chunk #{index} has unknown truth tags: {sorted(unknown_truth_tags)}")
        unknown_phase_tags = set(chunk["phase_tags"]) - known_phase_tags
        if unknown_phase_tags:
            raise RuntimeError(f"Chunk #{index} has unknown phase tags: {sorted(unknown_phase_tags)}")

    print("Retrieval preparation validation passed.")
    print(f"- Documents: {len(written['documents'])}")
    print(f"- Chunks: {len(written['chunks'])}")
    print(f"- Unique document ids: {len(set(document_ids))}")
    print(f"- Unique chunk ids: {len(set(chunk_ids))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
