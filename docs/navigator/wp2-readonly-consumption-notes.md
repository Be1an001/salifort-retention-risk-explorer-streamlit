# WP2 Read-only Consumption Notes

WP2 adds a small internal code layer for consuming the navigator registries introduced in WP1.

Scope of this layer:

- load the registry files with repo-relative paths
- validate them enough for safe programmatic use
- expose deterministic helper/query functions
- support future navigator UI, RAG indexing, and offline orchestration planning

Intentional non-goals in WP2:

- no API integration
- no Streamlit page or routing changes
- no artifact or builder rewrites
- no runtime retraining
- no repo rename

Implementation notes:

- The loader uses only the Python standard library.
- JSON registries are loaded with `json`.
- `navigator/source_registry.yaml` is parsed with a constrained, purpose-built parser for the exact YAML subset used in WP1. This avoids adding a new YAML dependency just to consume one controlled registry file.
- The new code lives under `app/services/` and is not wired into the current app runtime, so current user-facing behavior remains unchanged.

Near-term follow-on use:

- WP3 can import these modules to power a non-invasive navigator page.
- Later retrieval work can index registry content without guessing file semantics.
- Later orchestration work can use the governance summary and drift registry to keep automated workflows aligned with preserved truth rules.
