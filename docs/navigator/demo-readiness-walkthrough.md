# Demo Readiness Walkthrough

This walkthrough describes the current PACE Navigator demo posture and review boundaries.

## What To Show

1. Open the PACE Navigator page.
2. Start with **Final System Readiness** to explain what is ready, review-needed,
   blocked, or preview-only.
3. Show **Current Public Truth**:
   - public model truth: weighted XGBoost
   - selected threshold: 0.29
   - artifact-backed runtime remains the governed app posture
4. Use the **Governed Answer Viewer** with a fixed query such as:
   - `what is the public model truth`
   - `how is fallback different from final model truth`
   - `why is threshold 0.29 used`
5. Show the reviewer tabs:
   - support review
   - citations
   - source detail
   - audit checklist
   - export review
   - retrieval inspector
6. Show **Governed Orchestration Contracts** to explain task/workflow boundaries.
7. Show **Governed PACE Agent Shell** to demonstrate preview-only planning. The shell maps known request types to known workflow information; it does not execute work.

## What Requires API Key

Live retrieval-backed answer/reviewer paths require the user to provide an API
key manually through `RAG_STREAMLIT_OPENAI_API_KEY` or `OPENAI_API_KEY`.

No key should ever be committed into this repository.

## What Requires Offline CLI Use

Artifact-building and validation tasks remain CLI/offline tasks. Streamlit does
not execute workflows, trigger Airflow, or mutate repository files.

## What Is Intentionally Preview-Only

- Agent shell planning.
- Orchestration workflow inspection.
- Airflow scaffold visibility.
- Streamlit workflow readiness summaries.

## What Not To Claim

- Do not describe this as an autonomous HR decision engine.
- Do not say fallback heuristics are final model probabilities.
- Do not imply Airflow is production-deployed.
- Do not imply Streamlit can execute governed workflows.

## Demo-Ready Summary

The Navigator is ready to review as a governed project-inspection surface with
fixed-question retrieval support, explicit approval boundaries, and no
autonomous execution.
