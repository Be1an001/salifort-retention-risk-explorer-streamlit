# Navigator Notes

This folder documents the **PACE Navigator**, the advanced reviewer layer inside the portfolio app.

The Navigator starts as a guided project map and then offers optional tools for reviewers who want to inspect project truth, drift, retrieval evidence, source previews, audit exports, workflow contracts, readiness notes, and preview-only planning.

## When to Read These Notes

Start elsewhere first if you are new to the repo:

- [Root README](../../README.md)
- [Documentation Guide](../README.md)
- [User Manual](../user-guide/user-manual.md)
- [Streamlit App Walkthrough](../user-guide/streamlit-app-walkthrough.md)

Use this folder when you want a deeper technical review of the advanced Navigator layer.

## Recommended Reading Order

1. `demo-readiness-walkthrough.md` for the current demo posture and review boundaries.
2. `../technical/technical-design-and-architecture.md` for the system-level technical view.
3. The `wp*.md` note files in this folder if you want implementation-history detail for the Navigator build-out.

The `wp*.md` files preserve phase-time wording. When an older note says "future" or "later phases," read that relative to the work package date; current canonical behavior is summarized in this README, the root README, and the technical design document.

## Scope of These Notes

These notes help explain:

- how the Navigator organizes project truth and drift
- where retrieval-backed reviewer tools fit
- how workflow contracts and readiness summaries are modeled
- why the agent shell is preview-only
- why Airflow visibility in this repo is scaffold-level rather than runtime execution

The Navigator is not a chatbot, production scheduler, or autonomous agent. Its advanced features are meant to make the portfolio project easier to audit.
