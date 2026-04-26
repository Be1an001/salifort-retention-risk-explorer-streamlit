from __future__ import annotations

import ast
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.pages import mlops_lab

MLOPS_LAB_PAGE = REPO_ROOT / "app" / "pages" / "mlops_lab.py"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
MLOPS_REQUIREMENTS = REPO_ROOT / "requirements-mlops.txt"


def _page_text() -> str:
    return MLOPS_LAB_PAGE.read_text(encoding="utf-8")


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "employee_name": "Example Person",
                "email": "person@example.com",
                "satisfaction_level": 0.34,
                "last_evaluation": 0.91,
                "number_project": 6,
                "average_montly_hours": 246,
                "time_spend_company": 5,
                "Work_accident": 0,
                "left": 1,
                "promotion_last_5years": 0,
                "Department": "sales",
                "salary": "low",
            },
            {
                "employee_name": "Second Person",
                "email": "second@example.com",
                "satisfaction_level": 0.78,
                "last_evaluation": 0.71,
                "number_project": 3,
                "average_montly_hours": 168,
                "time_spend_company": 2,
                "Work_accident": 0,
                "left": 0,
                "promotion_last_5years": 0,
                "Department": "technical",
                "salary": "medium",
            },
        ]
    )


def test_mlops_lab_page_exists_and_has_mode_a_boundary_text() -> None:
    text = _page_text()

    assert MLOPS_LAB_PAGE.exists()
    assert "Online CSV Insight" in text
    assert "transparent review-priority heuristic" in text
    assert "raw CSV is not sent to the model" in text
    assert "not the public weighted XGBoost model probability" in text
    assert "not an employment decision" in text


def test_mlops_lab_has_no_shell_or_workflow_execution_calls() -> None:
    text = _page_text()

    forbidden = [
        "subprocess.run",
        "os.system",
        "Popen(",
        "airflow dags trigger",
        "git push",
        "docker compose up --",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_no_openai_call_at_module_import_level() -> None:
    tree = ast.parse(_page_text())
    top_level_calls = [
        node
        for statement in tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    rendered_attrs = {getattr(call.func, "attr", "") for call in top_level_calls}
    rendered_names = {getattr(call.func, "id", "") for call in top_level_calls}
    assert "create" not in rendered_attrs
    assert "_generate_ai_briefing" not in rendered_names


def test_mode_a_does_not_require_external_fastapi_secrets() -> None:
    text = _page_text()

    assert "SALIFORT_API_URL" not in text
    assert "SALIFORT_API_TOKEN" not in text
    assert "/batch-predict" not in text
    assert "Online API Scoring" not in text


def test_review_queue_excludes_pii_and_scores_heuristically() -> None:
    normalized, notes = mlops_lab.normalize_uploaded_columns(_sample_frame())
    valid, errors, quality = mlops_lab._validate_upload_frame(normalized)
    review_df = mlops_lab.build_review_queue(normalized)

    assert valid is True
    assert errors == []
    assert quality["left_present"] is True
    assert any("Identifier-like columns were excluded" in note for note in notes)
    assert "employee_name" not in review_df.columns
    assert "email" not in review_df.columns
    assert "left" not in review_df.columns
    assert {"uploaded_row_id", "review_score", "review_band", "review_reasons", "scoring_mode"}.issubset(
        review_df.columns
    )
    assert review_df["review_score"].between(0, 100).all()
    assert set(review_df["scoring_mode"]) == {"streamlit_heuristic"}


def test_sample_template_is_realistic_and_identifier_free() -> None:
    sample_df = pd.read_csv(StringIO(mlops_lab._synthetic_demo_csv()))

    assert len(sample_df) == 100
    assert set(mlops_lab.REQUIRED_FEATURE_COLUMNS).issubset(sample_df.columns)
    assert "left" not in sample_df.columns
    assert not any(mlops_lab._is_pii_like_column(column) for column in sample_df.columns)
    assert set(sample_df["department"]).issuperset(
        {"sales", "technical", "support", "IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng"}
    )
    assert set(sample_df["salary"]) == {"low", "medium", "high"}

    review_df = mlops_lab.build_review_queue(sample_df)
    band_counts = review_df["review_band"].value_counts().to_dict()
    assert {"High", "Medium", "Low"}.issubset(set(review_df["review_band"]))
    assert band_counts["High"] >= 20
    assert band_counts["Medium"] >= 20
    assert band_counts["Low"] >= 20


def test_minimal_schema_template_is_small_and_identifier_free() -> None:
    template_df = pd.read_csv(StringIO(mlops_lab._minimal_template_csv()))

    assert 8 <= len(template_df) <= 12
    assert set(mlops_lab.REQUIRED_FEATURE_COLUMNS).issubset(template_df.columns)
    assert "left" not in template_df.columns
    assert not any(mlops_lab._is_pii_like_column(column) for column in template_df.columns)


def test_identifier_detection_uses_strict_column_matching() -> None:
    assert mlops_lab._is_pii_like_column("work_accident") is False
    assert mlops_lab._is_pii_like_column("uploaded_row_id") is False
    assert mlops_lab._is_pii_like_column("satisfaction_level") is False
    assert mlops_lab._is_pii_like_column("employee_id") is True
    assert mlops_lab._is_pii_like_column("email") is True
    assert mlops_lab._is_pii_like_column("name") is True
    assert mlops_lab._is_pii_like_column("phone") is True
    assert mlops_lab._is_pii_like_column("address") is True
    assert mlops_lab._is_pii_like_column("personal_email") is True
    assert mlops_lab._is_pii_like_column("mobile") is True
    assert mlops_lab._is_pii_like_column("national_id") is True
    assert mlops_lab._is_pii_like_column("row_id") is True


def test_compact_openai_summary_excludes_raw_rows_and_pii() -> None:
    normalized, notes = mlops_lab.normalize_uploaded_columns(_sample_frame())
    _, _, quality = mlops_lab._validate_upload_frame(normalized)
    review_df = mlops_lab.build_review_queue(normalized)
    compact = mlops_lab.build_compact_openai_summary(review_df, quality, notes)

    serialized = str(compact)
    assert compact["scoring_mode"] == "streamlit_heuristic"
    assert "employee_name" not in serialized
    assert "person@example.com" not in serialized
    assert "email" not in serialized
    assert "top_priority_rows" in compact
    assert "department_review_summary" in compact
    assert "recommended_review_queue" in compact
    assert len(compact["top_priority_rows"]) <= 10


def test_top_departments_exclude_zero_high_counts() -> None:
    sample_df = pd.read_csv(StringIO(mlops_lab._synthetic_demo_csv()))
    review_df = mlops_lab.build_review_queue(sample_df)
    compact = mlops_lab.build_compact_openai_summary(review_df, {"uploaded_rows": len(review_df)}, [])

    assert compact["top_departments_by_high_count"]
    assert all(item["high_count"] > 0 for item in compact["top_departments_by_high_count"])


def test_deterministic_insight_pack_includes_row_and_department_recommendations() -> None:
    sample_df = pd.read_csv(StringIO(mlops_lab._synthetic_demo_csv()))
    review_df = mlops_lab.build_review_queue(sample_df)
    insights = mlops_lab.build_deterministic_insight_pack(review_df)

    assert insights["top_priority_rows"]
    assert insights["department_review_summary"]
    assert insights["recommended_review_queue"]["first_priority_row_ids"]
    assert insights["recommended_review_queue"]["departments_to_review"]
    assert all(item["high_count"] > 0 for item in insights["top_departments_by_high_count"])


def test_requirements_define_openai_only_for_streamlit_runtime() -> None:
    runtime_requirements = REQUIREMENTS.read_text(encoding="utf-8")
    mlops_requirements = MLOPS_REQUIREMENTS.read_text(encoding="utf-8")

    assert "openai>=2,<3" in runtime_requirements
    assert "openai" not in mlops_requirements.lower()
