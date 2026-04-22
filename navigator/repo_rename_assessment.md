# Repo Rename Assessment

## Current Repo Name Evaluation

Current name: `salifort-retention-risk-explorer-streamlit`

The current name is still accurate for the repo as it exists today. It clearly signals:

- Salifort project scope
- retention-risk focus
- explorer-style app framing
- Streamlit implementation layer

That makes it acceptable for current public portfolio use, current deployment clarity, and GitHub discoverability.

## Pros Of Keeping The Current Name

- It matches the actual current repo role: the public Streamlit app layer.
- It stays aligned with the deployed app and current README framing.
- It avoids unnecessary rename churn before navigator, orchestration, or multi-surface capabilities actually exist.
- It preserves continuity for existing links, portfolio references, and Streamlit deployment expectations.

## Reasons To Rename Later

A rename may become appropriate later if this repository expands beyond being primarily the Streamlit app layer and becomes the broader system-of-record for:

- PACE navigator registries
- source-of-truth governance
- future RAG indexing
- future agent orchestration
- future Airflow offline orchestration
- possibly multiple app surfaces beyond Streamlit

If that scope expansion happens, the current name may become too implementation-specific and too narrow.

## Recommended Future Name Candidates

If the repo later becomes the broader system hub, the strongest future candidate is:

1. `salifort-pace-analytics-navigator`

Additional acceptable candidates:

1. `salifort-pace-analytics-system`
2. `salifort-pace-system-navigator`

Why `salifort-pace-analytics-navigator` is the best candidate:

- it preserves Salifort project identity
- it reflects the future PACE method spine
- it is broader than Streamlit but not prematurely agent-branded
- it remains suitable for portfolio, GitHub, and system-navigation use

## Public Portfolio Implications

For the current public portfolio state, the existing name is still strong because it tells reviewers exactly what they are opening.

If the repo later becomes a wider platform or governance hub, a future rename would improve clarity by signaling that the repo is no longer just the Streamlit explorer.

## Deployment / Streamlit / GitHub Implications

Keeping the current name now reduces short-term friction:

- fewer broken GitHub links
- fewer deployment-name mismatches
- fewer README and badge updates
- less confusion between repo identity and deployed app identity

A later rename would likely require coordinated updates to:

- GitHub remote URLs
- README links
- deployment references
- portfolio references
- any external documentation or pinned links

## Migration Risks

- broken inbound GitHub links
- stale clone instructions
- mismatch between repo name and existing deployed app branding
- confusion if rename happens before the broader navigator system is actually present
- possible accidental scope inflation in public messaging

## Final Recommendation For Now

Do not rename the repo during this recommendation-only assessment.

Recommendation for now: keep `salifort-retention-risk-explorer-streamlit` as-is.

Recommendation for later: reassess renaming only after the repo genuinely becomes the broader PACE navigator / governance / orchestration hub. If that threshold is crossed, prefer `salifort-pace-analytics-navigator`.
