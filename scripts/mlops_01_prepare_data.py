from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salifort_mlops.config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    LAB_REPORTS_DIR,
)
from salifort_mlops.data_prep import (  # noqa: E402
    build_data_profile,
    clean_salifort_data,
    ensure_directories,
    load_raw_data,
    split_train_test,
)
from salifort_mlops.features import build_hr_features  # noqa: E402
from salifort_mlops.schemas import validate_category_values, validate_columns  # noqa: E402


DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "hr_capstone_dataset.csv"
TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "test.parquet"
PROFILE_OUTPUT_PATH = LAB_REPORTS_DIR / "data_profile.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Salifort MLOps Mini-Lab data without training models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw Salifort CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["operational", "survey_rich"],
        default="operational",
        help="Feature engineering mode.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Stratified test split fraction.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def run_prepare_data(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    mode: str = "operational",
    test_size: float = 0.2,
) -> dict[str, Any]:
    raw_df = load_raw_data(input_path)
    raw_validation = validate_columns(raw_df, allow_raw_aliases=True)
    raw_validation.raise_for_errors()

    clean_df = clean_salifort_data(raw_df)
    category_validation = validate_category_values(clean_df)
    category_validation.raise_for_errors()

    feature_df = build_hr_features(clean_df, mode=mode)
    train_df, test_df = split_train_test(
        feature_df,
        test_size=test_size,
        random_state=RANDOM_SEED,
    )

    raw_profile = build_data_profile(raw_df)
    clean_profile = build_data_profile(clean_df)
    feature_profile = build_data_profile(feature_df)
    report = {
        "pipeline": "mlops_01_prepare_data",
        "mode": mode,
        "input_path": str(input_path),
        "raw": raw_profile,
        "clean": clean_profile,
        "features": feature_profile,
        "duplicates_removed": raw_profile["row_count"] - clean_profile["row_count"],
        "split": {
            "test_size": test_size,
            "random_state": RANDOM_SEED,
            "stratified_on": TARGET_COLUMN,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_target_counts": {
                str(key): int(value)
                for key, value in train_df[TARGET_COLUMN]
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
                .items()
            },
            "test_target_counts": {
                str(key): int(value)
                for key, value in test_df[TARGET_COLUMN]
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
                .items()
            },
        },
        "outputs": {
            "train": str(TRAIN_OUTPUT_PATH),
            "test": str(TEST_OUTPUT_PATH),
            "data_profile": str(PROFILE_OUTPUT_PATH),
        },
    }

    ensure_directories()
    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_parquet(TEST_OUTPUT_PATH, index=False)
    PROFILE_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Salifort MLOps prepare-data")
    print(f"raw rows: {report['raw']['row_count']}")
    print(f"clean rows: {report['clean']['row_count']}")
    print(f"duplicates removed: {report['duplicates_removed']}")
    print(f"target counts: {report['clean'].get('target_counts', {})}")
    print(f"train rows: {report['split']['train_rows']}")
    print(f"test rows: {report['split']['test_rows']}")
    print(f"train output: {report['outputs']['train']}")
    print(f"test output: {report['outputs']['test']}")
    print(f"profile output: {report['outputs']['data_profile']}")


def main() -> int:
    args = parse_args()
    report = run_prepare_data(
        input_path=args.input,
        mode=args.mode,
        test_size=args.test_size,
    )
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
