from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salifort_mlops.config import (  # noqa: E402
    LEGACY_LAB_CHAMPION_MODEL_PATH,
    PROCESSED_DATA_DIR,
    STABLE_CHAMPION_MODEL_PATH,
)
from salifort_mlops.train import (  # noqa: E402
    log_training_to_mlflow,
    save_candidate_models,
    save_model_artifact,
    save_training_reports,
    train_candidate_models,
)

TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"


def _require_processed_data() -> None:
    missing = [path for path in (TRAIN_PATH, TEST_PATH) if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Processed data is missing: {paths}. "
            "Run python scripts/mlops_01_prepare_data.py first."
        )


def run_training() -> dict[str, object]:
    _require_processed_data()
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    training_result = train_candidate_models(train_df, test_df, mode="operational")
    model_paths = save_candidate_models(training_result)
    champion_name = str(training_result["champion"]["model_name"])
    save_model_artifact(
        training_result["fitted_models"][champion_name],
        LEGACY_LAB_CHAMPION_MODEL_PATH,
    )
    save_model_artifact(
        training_result["fitted_models"][champion_name],
        STABLE_CHAMPION_MODEL_PATH,
    )
    report_paths = save_training_reports(training_result=training_result)
    mlflow_summary = log_training_to_mlflow(
        training_result=training_result,
        model_paths=model_paths,
    )
    return {
        "training_result": training_result,
        "model_paths": model_paths,
        "report_paths": report_paths,
        "mlflow_summary": mlflow_summary,
    }


def print_summary(payload: dict[str, object]) -> None:
    training_result = payload["training_result"]
    champion = training_result["champion"]
    print("Salifort MLOps train-model")
    print(f"candidate models: {list(training_result['fitted_models'].keys())}")
    print(f"lab champion: {champion['model_name']}")
    print(f"best threshold: {float(champion['best_threshold']):.2f}")
    print(
        "champion metrics: "
        f"cost={champion['best_cost']}, "
        f"recall={float(champion['best_recall']):.3f}, "
        f"precision={float(champion['best_precision']):.3f}, "
        f"f2={float(champion['best_f2']):.3f}, "
        f"pr_auc={float(champion['best_pr_auc']):.3f}"
    )
    print(f"model outputs: {payload['model_paths']}")
    print(f"report outputs: {payload['report_paths']}")
    print(f"mlflow summary: {payload['mlflow_summary']}")
    print("This lab champion does not replace the public weighted XGBoost threshold 0.29 story.")


def main() -> int:
    payload = run_training()
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
