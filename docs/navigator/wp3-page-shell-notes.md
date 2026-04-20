# WP3 Page Shell Notes

WP3 adds the first non-invasive Streamlit shell for the future PACE Navigator.

What was added:

- a UI-facing read-model layer under `app/viewmodels/`
- a new page shell at `app/pages/pace_navigator.py`
- conservative navigation wiring so the page is visible in the existing app

Why this is still non-invasive:

- no API integration was introduced
- no runtime artifact logic changed
- no model or builder logic changed
- the page reads only local registry-backed view models
- the page is informational and deterministic, not a chat interface

What remains intentionally deferred:

- action buttons or guided workflows
- topic search input beyond deterministic examples
- retrieval/indexing integration
- API-backed explanations
- Airflow or orchestration integration

What later phases can build on:

- governed routing actions
- richer page recommendations
- source drilldowns
- future retrieval preparation
- future API-assisted navigator behavior once governance boundaries are ready
