# Navigator Metadata

This folder stores the structured metadata behind the PACE Navigator and advanced review tools.

Most visitors do not need to read these files directly. They are here so the app can keep project truth, drift notes, retrieval evidence, workflow contracts, and readiness rules explicit instead of burying them in page copy.

## Main Registry Files

- `source_registry.yaml`: inventory of important sources, artifacts, app pages, and presentation layers.
- `truth_registry.json`: canonical truth rules by project layer.
- `drift_register.json`: known differences between project layers that should stay visible.
- `pace_phase_map.json`: Plan, Analyze, Construct, Execute map for the project.
- `glossary.json`: shared definitions for important project terms.
- `repo_rename_assessment.md`: recommendation-only note; the repo has not been renamed.

## Advanced Review Areas

- `retrieval_pack/`: prepared documents and chunks for retrieval.
- `retrieval_index/`: local embedding index artifacts generated from the retrieval pack.
- `orchestration/`: workflow and task contracts for offline scheduling or orchestration review.
- `agent/`: controlled request catalog and plan-preview policy.
- `system/`: readiness and approval-gate metadata.

These files support review and demo transparency. They do not change the public model truth, retrain models, or make Streamlit execute workflows.

This folder helps the app explain where its claims come from, what evidence supports them, and which advanced features are review-only.

## Related Documents

- [Documentation Guide](../docs/README.md)
- [Technical Design and Architecture](../docs/technical/technical-design-and-architecture.md)
- [User Manual](../docs/user-guide/user-manual.md)
- [Navigator Notes](../docs/navigator/README.md)
