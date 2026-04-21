# WP11 Audit Workflow Notes

WP11 adds a controlled multi-query audit workflow to the existing PACE Navigator page.

What changed:

- added a fixed-query multiselect for reviewer audit workflows
- added an explicit opt-in control to run the multi-query workflow
- added per-query review cards and a cross-query comparison matrix
- added workflow summary counts for ready, blocked, strong-support, attention-needed, drift-heavy, and reference-supported queries
- added downloadable markdown, JSON, and text cross-query audit packets

Governance boundaries:

- the workflow only uses the existing fixed governed query set
- no free-text chat, unconstrained search, or generative rewrite was added
- blocked/no-key states remain visible and controlled
- public model truth, threshold truth, artifact truth, and fallback separation remain unchanged

Export behavior:

- combined packets include selected queries only
- packets preserve per-query direct answers, support-quality status, review notes, drift/caveats, citations, recommended pages, and retrieval-pack build context
- packets do not include API keys, environment values, secrets, or unsupported claims

Still deferred:

- no persisted reviewer notes
- no editable audit state
- no database storage
- no Airflow, agents, or background workflow execution
