# WP6 Embedding Retrieval Notes

WP6 adds the first API-backed retrieval layer for the Salifort PACE Navigator.

What WP6 adds:

- an offline embedding-index build script
- a local file-based retrieval index design
- a governed retriever service that loads the local index and returns traceable top-k chunks
- a retrieval validation script with fixed project-specific queries

How to provide the API key manually:

- set `RAG_STREAMLIT_OPENAI_API_KEY` in your local shell environment before running the build or validation scripts
- `OPENAI_API_KEY` is also accepted as a fallback
- do not commit any key, secret, `.env`, or populated secrets file into this repo

Default embedding model:

- `text-embedding-3-small` is the default embedding model because it is the lower-cost third-generation embedding option in OpenAI's official embeddings guide
- docs used for this choice:
  - [Embeddings API reference](https://platform.openai.com/docs/api-reference/embeddings/create?lang=python)
  - [Vector embeddings guide](https://platform.openai.com/docs/guides/embeddings)

How to build the local index:

- dry run without API call:
  - `python scripts/build_retrieval_index.py --dry-run`
- live build with a manually provided API key:
  - `python scripts/build_retrieval_index.py`

How to validate retrieval:

- dry run without live query embedding:
  - `python scripts/validate_retrieval_runtime.py --dry-run`
- live retrieval validation:
  - `python scripts/validate_retrieval_runtime.py`

What this phase still does not do:

- no answer generation
- no chat UI
- no retrieval-backed Streamlit assistant
- no vector database
- no orchestration or Airflow

Why this is still non-invasive:

- it does not change the business app logic
- it does not change model or artifact truth
- it keeps embedding/index work in offline scripts and local index artifacts
