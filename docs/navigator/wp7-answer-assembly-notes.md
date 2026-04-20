# WP7 Answer Assembly Notes

WP7 adds the first governed answer assembly layer on top of the existing retrieval runtime.

What this layer does:

- takes a user query
- runs governed retrieval against the existing local retrieval index
- assembles a deterministic structured answer object
- preserves truth, drift, route, and citation separation

What it does not do:

- no open-ended chat UI
- no generative answer synthesis
- no silent blending of fallback truth into final/public model truth
- no replacement of retrieval with opaque summarization

Why this is governance-safe:

- canonical truth chunks are prioritized for the answer headline and direct answer
- drift chunks are attached separately as caveats and handling context
- page recommendations are surfaced explicitly instead of being hidden inside prose
- citations and source paths are always carried through
- incomplete coverage is surfaced through `coverage_summary`, `governance_flags`, and `assembly_status`

What later phases can build on:

- a Streamlit retrieval inspector
- a governed answer viewer in the PACE Navigator
- optional API-backed explanation formatting above this deterministic layer
- future orchestration jobs that need structured governed summaries
