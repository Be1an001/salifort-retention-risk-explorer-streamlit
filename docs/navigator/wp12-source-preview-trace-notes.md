# WP12 Source Preview And Evidence Trace Notes

WP12 adds governed evidence tracing and eligible source-file preview to the PACE Navigator reviewer surface.

What changed:

- added an evidence trace chain for the selected governed chunk
- added explicit preview eligibility evaluation for each selected source path
- added read-only previews for small governed text-like files when eligible
- added preview eligibility metadata to single-query and cross-query audit exports

Preview boundary:

- this is not a generic repository browser
- only relative repo-local governed source paths are considered
- previews are capped by file size and character limits
- secret-like, environment-like, cache, local environment, binary, tabular, image, parquet, and vector files are blocked
- exports include preview metadata, not full source-preview text

Allowed preview extensions:

- `.md`
- `.json`
- `.txt`
- `.yaml`
- `.yml`
- `.toml`
- `.py`

Still deferred:

- no arbitrary source search
- no unrestricted full-file browsing
- no editable reviewer notes or persistence
- no agents, Airflow, or background workflow execution
