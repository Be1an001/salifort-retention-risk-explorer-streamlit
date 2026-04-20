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
    OpenAIEmbeddingConfig,
    build_retrieval_index,
)
from app.services.navigator_retrieval_pack import load_retrieval_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the governed local retrieval index.")
    parser.add_argument("--model", default=None, help="Override embedding model.")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional embedding dimensions override for text-embedding-3 models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional embeddings batch size override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pack availability and planned build inputs without calling the API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retrieval_pack = load_retrieval_pack()
    chunks = retrieval_pack["chunks"]

    if args.dry_run:
        print("Dry run: retrieval index build inputs validated.")
        print(f"- Retrieval pack chunks: {len(chunks)}")
        print(f"- Default embedding model: {args.model or 'text-embedding-3-small'}")
        print("- API call skipped by request.")
        return 0

    try:
        config = OpenAIEmbeddingConfig.from_env(
            embedding_model=args.model,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
        )
        result = build_retrieval_index(config=config)
    except NavigatorEmbeddingConfigurationError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except NavigatorEmbeddingRequestError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except NavigatorEmbeddingIndexError as exc:
        print(f"FAILED: {exc}")
        return 1

    print("Retrieval index built successfully.")
    print(f"- Output root: {result['output_root']}")
    print(f"- Chunks indexed: {result['chunk_count']}")
    print(f"- Vector count: {result['vector_count']}")
    print(f"- Vector dimensions: {result['vector_dimensions']}")
    print(f"- Embedding model: {result['manifest']['embedding_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
