from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from app.services.navigator_loader import get_repo_root
from app.services.navigator_retrieval_pack import get_retrieval_pack_root, load_retrieval_pack

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_BATCH_SIZE = 16
DEFAULT_API_KEY_ENV_VARS = ("RAG_STREAMLIT_OPENAI_API_KEY", "OPENAI_API_KEY")


class NavigatorEmbeddingIndexError(RuntimeError):
    """Base error for embedding-index build and load operations."""


class NavigatorEmbeddingConfigurationError(NavigatorEmbeddingIndexError):
    """Raised when the embedding configuration is incomplete."""


class NavigatorEmbeddingRequestError(NavigatorEmbeddingIndexError):
    """Raised when the OpenAI embeddings request fails."""


class NavigatorRetrievalIndexNotFoundError(NavigatorEmbeddingIndexError):
    """Raised when a built retrieval index cannot be found on disk."""


@dataclass(frozen=True)
class OpenAIEmbeddingConfig:
    api_key: str
    api_key_env_name: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    base_url: str = DEFAULT_OPENAI_BASE_URL
    dimensions: int | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE

    @classmethod
    def from_env(
        cls,
        *,
        embedding_model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
    ) -> "OpenAIEmbeddingConfig":
        api_key = None
        api_key_env_name = None
        for env_name in DEFAULT_API_KEY_ENV_VARS:
            env_value = os.environ.get(env_name)
            if env_value:
                api_key = env_value
                api_key_env_name = env_name
                break

        if not api_key or not api_key_env_name:
            raise NavigatorEmbeddingConfigurationError(
                "OpenAI API key not found. Set RAG_STREAMLIT_OPENAI_API_KEY "
                "or OPENAI_API_KEY manually in your local environment."
            )

        configured_dimensions = dimensions
        if configured_dimensions is None:
            raw_dimensions = os.environ.get("RAG_STREAMLIT_OPENAI_DIMENSIONS")
            if raw_dimensions:
                try:
                    configured_dimensions = int(raw_dimensions)
                except ValueError as exc:
                    raise NavigatorEmbeddingConfigurationError(
                        "RAG_STREAMLIT_OPENAI_DIMENSIONS must be an integer when provided."
                    ) from exc

        configured_model = (
            embedding_model
            or os.environ.get("RAG_STREAMLIT_OPENAI_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        configured_base_url = (
            base_url
            or os.environ.get("RAG_STREAMLIT_OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        configured_timeout = timeout_seconds or int(
            os.environ.get("RAG_STREAMLIT_OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        configured_batch_size = batch_size or int(
            os.environ.get("RAG_STREAMLIT_OPENAI_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        )

        return cls(
            api_key=api_key,
            api_key_env_name=api_key_env_name,
            embedding_model=configured_model,
            base_url=configured_base_url,
            dimensions=configured_dimensions,
            timeout_seconds=configured_timeout,
            batch_size=configured_batch_size,
        )


def get_retrieval_index_root() -> Path:
    return get_repo_root() / "navigator" / "retrieval_index"


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise NavigatorEmbeddingIndexError(
            "numpy is required for the local retrieval index. "
            "Install the repo runtime dependencies before building the index."
        ) from exc
    return np


def _compose_embedding_input(chunk: dict[str, Any]) -> str:
    return f"{chunk['title']}\n\n{chunk['text']}".strip()


def _preview_text(text: str, *, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _embed_text_batch(
    texts: list[str],
    *,
    config: OpenAIEmbeddingConfig,
) -> tuple[list[list[float]], dict[str, int]]:
    payload: dict[str, Any] = {
        "model": config.embedding_model,
        "input": texts,
        "encoding_format": "float",
    }
    if config.dimensions is not None:
        payload["dimensions"] = config.dimensions

    body = json.dumps(payload).encode("utf-8")
    api_request = request.Request(
        url=f"{config.base_url}/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(api_request, timeout=config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise NavigatorEmbeddingRequestError(
            f"OpenAI embeddings request failed with HTTP {exc.code}: {details}"
        ) from exc
    except error.URLError as exc:
        raise NavigatorEmbeddingRequestError(
            f"OpenAI embeddings request failed: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise NavigatorEmbeddingRequestError(
            f"OpenAI embeddings request failed: {exc}"
        ) from exc

    payload = json.loads(response_body)
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise NavigatorEmbeddingRequestError(
            "OpenAI embeddings response did not match the requested batch size."
        )

    vectors = [item["embedding"] for item in data]
    usage = payload.get("usage") or {}
    usage_summary = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return vectors, usage_summary


def _normalize_vectors(vectors):
    np = _require_numpy()
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise NavigatorEmbeddingIndexError("Embedding matrix must be two-dimensional.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _build_chunk_index_records(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_kind": chunk["chunk_kind"],
                "title": chunk["title"],
                "text_preview": _preview_text(chunk["text"]),
                "source_paths": chunk["source_paths"],
                "registry_refs": chunk["registry_refs"],
                "truth_tags": chunk["truth_tags"],
                "drift_tags": chunk["drift_tags"],
                "phase_tags": chunk["phase_tags"],
                "page_routes": chunk["page_routes"],
                "authority_level": chunk["authority_level"],
                "retrieval_role": chunk["retrieval_role"],
                "caveats": chunk["caveats"],
            }
        )
    return records


def build_retrieval_index(
    *,
    config: OpenAIEmbeddingConfig,
    output_root: Path | None = None,
) -> dict[str, Any]:
    np = _require_numpy()
    retrieval_pack = load_retrieval_pack()
    chunks = retrieval_pack["chunks"]
    if not chunks:
        raise NavigatorEmbeddingIndexError("Retrieval pack contains no chunks to embed.")

    texts = [_compose_embedding_input(chunk) for chunk in chunks]
    raw_vectors: list[list[float]] = []
    prompt_tokens = 0
    total_tokens = 0
    request_count = 0

    for start_index in range(0, len(texts), config.batch_size):
        batch = texts[start_index : start_index + config.batch_size]
        batch_vectors, usage = _embed_text_batch(batch, config=config)
        raw_vectors.extend(batch_vectors)
        prompt_tokens += usage["prompt_tokens"]
        total_tokens += usage["total_tokens"]
        request_count += 1

    normalized_vectors = _normalize_vectors(raw_vectors)
    vector_dimensions = int(normalized_vectors.shape[1])

    root = output_root or get_retrieval_index_root()
    root.mkdir(parents=True, exist_ok=True)

    chunk_index_records = _build_chunk_index_records(chunks)
    vector_path = root / "chunk_vectors.npy"
    index_path = root / "chunk_index.jsonl"
    manifest_path = root / "manifest.json"

    np.save(vector_path, normalized_vectors)
    with index_path.open("w", encoding="utf-8") as handle:
        for record in chunk_index_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    retrieval_pack_root = get_retrieval_pack_root()
    retrieval_manifest_path = retrieval_pack_root / "manifest.json"
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "index_version": "1.0",
        "embedding_provider": "OpenAI",
        "embedding_model": config.embedding_model,
        "embedding_dimensions_requested": config.dimensions,
        "embedding_dimensions_resolved": vector_dimensions,
        "embedding_input_policy": "Chunk title plus chunk text, normalized into unit-length float32 vectors.",
        "retrieval_pack_manifest": str(retrieval_manifest_path.relative_to(get_repo_root())),
        "retrieval_pack_version": retrieval_manifest.get("pack_version"),
        "retrieval_pack_chunk_count": retrieval_manifest.get("chunk_count"),
        "retrieval_pack_document_count": retrieval_manifest.get("document_count"),
        "index_chunk_count": len(chunk_index_records),
        "index_vector_count": int(normalized_vectors.shape[0]),
        "vector_storage": "navigator/retrieval_index/chunk_vectors.npy",
        "index_storage": "navigator/retrieval_index/chunk_index.jsonl",
        "normalized_vectors": True,
        "vector_dtype": "float32",
        "api_key_env_name": config.api_key_env_name,
        "build_request_count": request_count,
        "build_prompt_tokens": prompt_tokens,
        "build_total_tokens": total_tokens,
        "supported_filters": [
            "truth_tag",
            "phase_tag",
            "retrieval_role",
            "source_path_contains",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "output_root": root,
        "manifest": manifest,
        "chunk_count": len(chunk_index_records),
        "vector_count": int(normalized_vectors.shape[0]),
        "vector_dimensions": vector_dimensions,
    }


def load_retrieval_index(output_root: Path | None = None) -> dict[str, Any]:
    np = _require_numpy()
    root = output_root or get_retrieval_index_root()
    manifest_path = root / "manifest.json"
    index_path = root / "chunk_index.jsonl"
    vector_path = root / "chunk_vectors.npy"

    missing_files = [
        str(path.relative_to(get_repo_root()))
        for path in (manifest_path, index_path, vector_path)
        if not path.exists()
    ]
    if missing_files:
        raise NavigatorRetrievalIndexNotFoundError(
            "Retrieval index is incomplete or missing: " + ", ".join(missing_files)
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_index = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(vector_path)
    return {
        "root": root,
        "manifest": manifest,
        "chunk_index": chunk_index,
        "vectors": vectors,
    }
