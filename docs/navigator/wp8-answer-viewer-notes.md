# WP8 Answer Viewer Notes

WP8 adds the first user-visible governed retrieval and answer-viewer surface inside the existing PACE Navigator page.

What users can do:

- select one of the fixed governed query scenarios
- view the assembled answer title and direct answer
- inspect coverage status and governance flags
- review supporting truth separately from drift and caveats
- inspect recommended pages, citations, and raw retrieval results

Why this remains governed:

- the primary interaction is a fixed controlled query selector
- answer assembly is still deterministic and non-generative
- drift stays separate from the direct answer
- low coverage or blocked states are shown explicitly instead of hidden

What this phase still does not do:

- no free-text chat
- no open-ended assistant
- no answer generation model
- no changes to core model or artifact truth
