from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salifort_mlops.config import DEFAULT_THRESHOLD, LAB_REPORTS_DIR  # noqa: E402
from salifort_mlops.train import select_lab_champion  # noqa: E402

TRAINING_RESULTS_PATH = LAB_REPORTS_DIR / "training_results.csv"
EVALUATION_SUMMARY_PATH = LAB_REPORTS_DIR / "evaluation_summary.json"
MODEL_CARD_PATH = LAB_REPORTS_DIR / "model_card.md"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _require_training_results() -> None:
    if not TRAINING_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Training results not found at {TRAINING_RESULTS_PATH}. "
            "Run python scripts/mlops_02_train_model.py first."
        )


def run_evaluation_summary() -> dict[str, Any]:
    _require_training_results()
    results_df = pd.read_csv(TRAINING_RESULTS_PATH)
    champion = select_lab_champion(results_df)
    summary = {
        "scope": "mlops-mini-lab",
        "public_model_replacement": False,
        "public_reference_model": "weighted XGBoost",
        "public_reference_threshold": DEFAULT_THRESHOLD,
        "lab_champion": champion,
        "selection_rule": "lowest best_cost, then highest best_recall, best_f2, and best_precision",
    }
    LAB_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    MODEL_CARD_PATH.write_text(_build_model_card(summary), encoding="utf-8")
    return summary


def _build_model_card(summary: dict[str, Any]) -> str:
    champion = summary["lab_champion"]
    return (
        "# Salifort MLOps Mini-Lab Model Card\n\n"
        "This model card summarizes a local/dev lab run. It does not replace "
        "the public artifact-backed Streamlit model truth.\n\n"
        "## Public App Boundary\n\n"
        "- Public reference model: weighted XGBoost\n"
        f"- Public selected threshold: {summary['public_reference_threshold']}\n"
        "- Public artifacts remain under `artifacts/v2/` and are not modified by this lab.\n\n"
        "## Lab Champion\n\n"
        f"- Model: {champion['model_name']}\n"
        f"- Tuned lab threshold: {float(champion['best_threshold']):.2f}\n"
        f"- Cost: {champion['best_cost']}\n"
        f"- Recall: {float(champion['best_recall']):.4f}\n"
        f"- Precision: {float(champion['best_precision']):.4f}\n"
        f"- F2: {float(champion['best_f2']):.4f}\n"
        f"- PR AUC: {float(champion['best_pr_auc']):.4f}\n\n"
        "## Responsible Use\n\n"
        "This lab output is for portfolio MLOps demonstration and review support only. "
        "It must not be used for automated employment decisions.\n"
    )


def print_summary(summary: dict[str, Any]) -> None:
    champion = summary["lab_champion"]
    print("Salifort MLOps evaluate-model")
    print(f"lab champion: {champion['model_name']}")
    print(f"best threshold: {float(champion['best_threshold']):.2f}")
    print(f"evaluation summary: {EVALUATION_SUMMARY_PATH}")
    print(f"model card: {MODEL_CARD_PATH}")
    print("Public app truth remains weighted XGBoost at threshold 0.29.")


def main() -> int:
    summary = run_evaluation_summary()
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
