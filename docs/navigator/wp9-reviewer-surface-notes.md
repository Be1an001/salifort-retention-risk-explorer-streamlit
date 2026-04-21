# WP9 Reviewer Surface Notes

WP9 upgrades the PACE Navigator answer viewer into a reviewer-grade inspection surface.

What changed:

- added controlled top-k retrieval depth for the fixed governed query set
- added reviewer filters for truth tags, drift tags, phase tags, retrieval role, page route, and authority level
- added deterministic sorting modes for similarity, authority priority, truth-first, drift-first, and retrieval-role grouping
- added side-by-side citation comparison with chunk metadata, provenance, caveats, preview text, and full chunk text
- added support-quality review indicators that call out canonical truth, drift context, page-route support, and reference-only support

What remains governed:

- no free-text chat
- no unconstrained answer generation
- no changes to public model truth, threshold truth, artifact truth, or fallback separation
- no hidden handling of missing API keys or missing retrieval index files

What remains deferred:

- no source-file content browser beyond governed chunk text
- no answer-generation model
- no agent, Airflow, or background orchestration layer
- no unrestricted query entry point
