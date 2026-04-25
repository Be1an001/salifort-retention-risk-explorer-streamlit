"""Central configuration for the Salifort MLOps Mini-Lab.

The constants here mirror the current public app truth without replacing the
artifact-backed Streamlit runtime. Future lab models should write to the lab
folders defined below, not to ``artifacts/v2``.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

REFERENCES_SOURCE_WORKFLOW_DIR = PROJECT_ROOT / "references" / "source_workflow"
SOURCE_WORKFLOW_DIR = PROJECT_ROOT / "source_workflow"
DATA_DIR = PROJECT_ROOT / "data"
MLOPS_ROOT = PROJECT_ROOT / "mlops"
PROCESSED_DATA_DIR = MLOPS_ROOT / "data" / "processed"
LAB_MODELS_DIR = MLOPS_ROOT / "models"
LAB_REPORTS_DIR = MLOPS_ROOT / "reports"

RAW_DATA_CANDIDATE_PATHS = (
    REFERENCES_SOURCE_WORKFLOW_DIR / "hr_capstone_dataset.csv",
    SOURCE_WORKFLOW_DIR / "hr_capstone_dataset.csv",
    DATA_DIR / "hr_capstone_dataset.csv",
)

TARGET_COLUMN = "left"
DEFAULT_THRESHOLD = 0.29
RANDOM_SEED = 42

FALSE_NEGATIVE_COST = 8.0
FALSE_POSITIVE_COST = 1.0

RAW_TO_NORMALIZED_COLUMNS = {
    "Work_accident": "work_accident",
    "average_montly_hours": "average_monthly_hours",
    "time_spend_company": "tenure",
    "Department": "department",
}

ALLOWED_SALARY_VALUES = ("low", "medium", "high")
ALLOWED_DEPARTMENT_VALUES = (
    "IT",
    "RandD",
    "accounting",
    "hr",
    "management",
    "marketing",
    "product_mng",
    "sales",
    "support",
    "technical",
)

SALARY_LEVEL_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

SALARY_COST_INDEX = {
    "low": 1.0,
    "medium": 1.3,
    "high": 1.7,
}

MODEL_MODES = ("operational", "survey_rich")
OPERATIONAL_EXCLUDED_FEATURES = (
    "satisfaction_level",
    "burnout_index",
    "effort_reward_gap",
    "low_satisfaction_high_eval",
)
