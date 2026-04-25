from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salifort_mlops.config import DEFAULT_THRESHOLD
from salifort_mlops.data_prep import build_data_profile, clean_salifort_data, load_raw_data
from salifort_mlops.features import build_hr_features, split_features_target
from salifort_mlops.train import build_preprocessing_plan, describe_candidate_models


def main() -> int:
    raw = load_raw_data(REPO_ROOT / "data" / "hr_capstone_dataset.csv")
    clean = clean_salifort_data(raw)
    features = build_hr_features(clean, mode="operational")
    X, y = split_features_target(clean, mode="operational")
    profile = build_data_profile(clean)
    plan = build_preprocessing_plan(mode="operational")
    models = describe_candidate_models()

    print("Salifort MLOps Phase 1 smoke check")
    print(f"rows_clean={profile['row_count']}")
    print(f"features_shape={features.shape}")
    print(f"X_shape={X.shape}")
    print(f"y_length={len(y)}")
    print(f"default_public_threshold={DEFAULT_THRESHOLD}")
    print(f"categorical_features={plan['categorical_features']}")
    print(f"candidate_models={[model['name'] for model in models]}")
    print(f"target_counts={profile.get('target_counts', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
