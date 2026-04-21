# Navigator Foundation

This folder stores deterministic registry files for the future Salifort PACE Analytics System Navigator.

Work Package 1 keeps these files read-only and out of the current Streamlit runtime path. They are intended to help later phases with source-of-truth resolution, drift explanation, phase mapping, repo governance, and future indexing or orchestration work.

Files in this folder:

- `source_registry.yaml`: inventory of the main sources, artifacts, and presentation layers.
- `truth_registry.json`: canonical truth rules by project layer.
- `drift_register.json`: known or confirmed drift items between layers.
- `pace_phase_map.json`: mapping of the Salifort project into the PACE structure.
- `glossary.json`: normalized definitions for key project terms.
- `repo_rename_assessment.md`: recommendation-only assessment of whether the repo should be renamed later.

Usage guidance for later phases:

- Treat these files as explicit planning inputs, not as runtime configuration.
- Preserve the distinction between public narrative truth, runtime artifact truth, legacy method truth, and fallback heuristic behavior.
- Extend entries by adding new records rather than overwriting prior reasoning without evidence.
