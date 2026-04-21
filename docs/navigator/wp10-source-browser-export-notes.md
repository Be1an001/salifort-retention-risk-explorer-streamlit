# WP10 Source Browser And Export Notes

WP10 adds two reviewer-facing capabilities to the existing PACE Navigator page.

What changed:

- added a governed source-detail browser for selected citations and retrieved chunks
- added related-chunk context from the same governed document or source path
- added a copyable markdown review summary
- added downloadable markdown, JSON, and text review packets

Important boundary:

- the source browser is chunk-level, not unrestricted full-file browsing
- it exposes retrieval-pack context, source paths, registry refs, tags, authority, caveats, preview text, and full governed chunk text
- it does not pretend to show raw source files beyond the governed retrieval-pack content

Export behavior:

- export payloads include the selected query, direct governed answer, support-quality status, supporting points, drift/caveats, recommended pages, citations, selected source detail, and retrieval-pack build context
- exports do not include API keys, environment values, secrets, or unsupported claims
- exports are deterministic from the current governed answer view and selected source detail

Still deferred:

- no free-form source search
- no editable notes or persistence
- no agent, Airflow, or background workflow layer
- no LLM-generated rewrite of the governed review summary
