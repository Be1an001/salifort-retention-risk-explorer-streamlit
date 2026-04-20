from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.navigator_retrieval_pack import write_retrieval_pack


def main() -> int:
    result = write_retrieval_pack()
    print("Retrieval preparation pack built successfully.")
    print(f"- Output root: {result['output_root']}")
    print(f"- Documents: {result['document_count']}")
    print(f"- Chunks: {result['chunk_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
