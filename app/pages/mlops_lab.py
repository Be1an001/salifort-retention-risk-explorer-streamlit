from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLOPS_ROOT = PROJECT_ROOT / "mlops"
REPORTS_DIR = MLOPS_ROOT / "reports"
MODELS_DIR = MLOPS_ROOT / "models"
PROCESSED_DIR = MLOPS_ROOT / "data" / "processed"

PUBLIC_REFERENCE_NOTE = (
    "The public app remains artifact-backed with weighted XGBoost at threshold 0.29. "
    "The MLOps Mini-Lab and hosted CSV Insight sandbox do not replace that public model truth."
)
RESPONSIBLE_USE_NOTE = (
    "Portfolio demonstration and human review support only. This page is not an employment decision system."
)
HOSTED_DEMO_NOTE = (
    "Hosted demo note: local FastAPI, MLflow, Docker Compose, Airflow, and generated lab model "
    "artifacts are local/dev components. They are not expected to be running inside the hosted "
    "Streamlit app. The Online CSV Insight sandbox runs directly in Streamlit Cloud without "
    "FastAPI, Docker, MLflow, Airflow, or generated model artifacts."
)
ONLINE_SANDBOX_NOTE = (
    "This online sandbox lets visitors upload a small Salifort-style CSV and receive a "
    "review-priority summary directly in Streamlit Cloud. The optional OpenAI briefing is "
    "generated from compact aggregate statistics only; the raw CSV is not sent to the model."
)
HEURISTIC_BOUNDARY_NOTE = (
    "This sandbox score is a transparent review-priority heuristic, "
    "not the public weighted XGBoost model probability and not an employment decision."
)

RAW_RENAME_MAP = {
    "average_montly_hours": "average_monthly_hours",
    "time_spend_company": "tenure",
    "Work_accident": "work_accident",
    "Department": "department",
}
REQUIRED_FEATURE_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    "promotion_last_5years",
    "department",
    "salary",
]
NUMERIC_FEATURE_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    "promotion_last_5years",
]
DOWNLOAD_COLUMNS = REQUIRED_FEATURE_COLUMNS + [
    "uploaded_row_id",
    "project_intensity",
    "review_score",
    "review_band",
    "review_reasons",
    "scoring_mode",
]
ALLOWED_SALARY_VALUES = {"low", "medium", "high"}
ALLOWED_DEPARTMENT_VALUES = {
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
}
PII_COLUMN_NAMES = {
    "id",
    "employee_id",
    "emp_id",
    "user_id",
    "staff_id",
    "row_id",
    "name",
    "employee_name",
    "full_name",
    "email",
    "personal_email",
    "phone",
    "address",
    "mobile",
    "national_id",
}
MAX_UPLOAD_ROWS = 2_000
SCORING_MODE = "streamlit_heuristic"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_text(path: Path, *, limit: int = 5000) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return ""


def _config_value(name: str, default: str = "") -> str:
    """Read Streamlit secrets first, then environment variables."""

    value: Any = None
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value in (None, ""):
        value = os.getenv(name, default)
    return str(value).strip() if value not in (None, "") else default


def _file_status_rows() -> list[dict[str, str]]:
    tracked = [
        ("Train split", PROCESSED_DIR / "train.parquet"),
        ("Test split", PROCESSED_DIR / "test.parquet"),
        ("Data profile", REPORTS_DIR / "data_profile.json"),
        ("Training results", REPORTS_DIR / "training_results.json"),
        ("Evaluation summary", REPORTS_DIR / "evaluation_summary.json"),
        ("Model card", REPORTS_DIR / "model_card.md"),
        ("Stable champion model", MODELS_DIR / "champion_model.joblib"),
        ("Legacy lab champion model", MODELS_DIR / "lab_champion.joblib"),
    ]
    rows = []
    for label, path in tracked:
        rows.append(
            {
                "Item": label,
                "Status": "Present" if path.exists() else "Missing",
                "Path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def _lab_champion() -> dict[str, Any]:
    evaluation_summary = _read_json(REPORTS_DIR / "evaluation_summary.json")
    if evaluation_summary.get("lab_champion"):
        return evaluation_summary["lab_champion"]
    training_results = _read_json(REPORTS_DIR / "training_results.json")
    if training_results.get("lab_champion"):
        return training_results["lab_champion"]
    return {}


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "Not available"


def _api_get(base_url: str, path: str) -> tuple[bool, dict[str, Any] | str]:
    url = base_url.rstrip("/") + path
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2) as response:
            payload = response.read().decode("utf-8")
            return True, json.loads(payload) if payload else {}
    except HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return False, str(exc)


def _status_badge(label: str, present: bool) -> None:
    if present:
        st.success(f"{label}: present")
    else:
        st.warning(f"{label}: missing")


def _code_block(command: str) -> None:
    st.code(command, language="bash")


def _is_pii_like_column(column: str) -> bool:
    normalized = column.lower().replace("-", "_").replace(" ", "_")
    return normalized in PII_COLUMN_NAMES


def normalize_uploaded_columns(upload_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize uploaded Salifort column names without mutating the input frame."""

    normalized = upload_df.rename(columns=RAW_RENAME_MAP).copy()
    notes: list[str] = []
    renamed = [f"{raw} -> {new}" for raw, new in RAW_RENAME_MAP.items() if raw in upload_df.columns]
    if renamed:
        notes.append("Normalized legacy columns: " + ", ".join(renamed))
    pii_like = [column for column in normalized.columns if _is_pii_like_column(column)]
    if pii_like:
        notes.append(
            "Identifier-like columns were excluded from displayed/downloaded review summaries "
            "and from the OpenAI briefing payload: "
            + ", ".join(sorted(pii_like))
        )
    if "left" in normalized.columns:
        notes.append("Target column `left` is used only for observed upload summaries, not scoring.")
    return normalized, notes


def _minimal_template_csv() -> str:
    sample_rows = [
        [0.38, 0.86, 5, 230, 5, 0, 0, "sales", "low"],
        [0.72, 0.77, 3, 172, 3, 0, 0, "technical", "medium"],
        [0.29, 0.91, 6, 255, 5, 0, 0, "support", "low"],
        [0.84, 0.62, 2, 145, 2, 1, 1, "management", "high"],
        [0.45, 0.82, 5, 214, 4, 0, 0, "product_mng", "medium"],
        [0.66, 0.74, 4, 188, 6, 0, 0, "RandD", "medium"],
        [0.52, 0.88, 5, 222, 4, 0, 0, "IT", "low"],
        [0.91, 0.57, 2, 132, 2, 0, 0, "marketing", "medium"],
    ]
    sample_df = pd.DataFrame(sample_rows, columns=REQUIRED_FEATURE_COLUMNS)
    return sample_df.to_csv(index=False)


def _synthetic_demo_csv() -> str:
    """Return a deterministic 100-row, PII-free scenario demo CSV."""

    departments = ["sales", "technical", "support", "IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng"]
    scenario_rows: list[list[Any]] = []
    for index in range(100):
        department = departments[index % len(departments)]
        scenario = index % 5
        if scenario == 0:  # stable low-review
            row = [0.82 - (index % 3) * 0.03, 0.64 + (index % 4) * 0.03, 3, 158 + (index % 5) * 4, 2 + (index % 3), index % 2, 1 if index % 10 == 0 else 0, department, "medium" if index % 4 else "high"]
        elif scenario == 1:  # workload-heavy
            row = [0.50 - (index % 4) * 0.03, 0.82 + (index % 3) * 0.03, 5 + (index % 2), 222 + (index % 6) * 5, 4 + (index % 3), 0, 0, department, "low" if index % 3 else "medium"]
        elif scenario == 2:  # longer tenure no promotion
            row = [0.58 - (index % 3) * 0.02, 0.73 + (index % 3) * 0.02, 5, 220 + (index % 4) * 3, 5 + (index % 3), 0, 0, department, "medium"]
        elif scenario == 3:  # high-evaluation overloaded
            row = [0.42 - (index % 3) * 0.04, 0.88 + (index % 3) * 0.03, 6, 238 + (index % 5) * 4, 4 + (index % 3), 0, 0, department, "low"]
        else:  # mixed medium-review
            row = [0.64 - (index % 2) * 0.03, 0.74 + (index % 3) * 0.02, 4, 220 + (index % 4) * 2, 4 + (index % 2), 0, 0, department, "high" if index % 3 == 0 else "medium"]
        scenario_rows.append(row)
    demo_df = pd.DataFrame(scenario_rows, columns=REQUIRED_FEATURE_COLUMNS)
    return demo_df.to_csv(index=False)


def _sample_csv() -> str:
    """Backward-compatible alias for the richer synthetic demo CSV."""

    return _synthetic_demo_csv()


def _validate_upload_frame(normalized: pd.DataFrame) -> tuple[bool, list[str], dict[str, Any]]:
    missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in normalized.columns]
    unexpected = [
        column
        for column in normalized.columns
        if column not in set(REQUIRED_FEATURE_COLUMNS + ["left"]) and not _is_pii_like_column(column)
    ]
    missing_counts = {
        column: int(normalized[column].isna().sum())
        for column in normalized.columns
        if column in set(REQUIRED_FEATURE_COLUMNS + ["left"]) and normalized[column].isna().sum() > 0
    }
    salary_issues: list[str] = []
    department_issues: list[str] = []
    if "salary" in normalized.columns:
        observed_salary = set(normalized["salary"].astype(str).str.strip().str.lower().dropna())
        salary_issues = sorted(value for value in observed_salary if value not in ALLOWED_SALARY_VALUES)
    if "department" in normalized.columns:
        observed_departments = set(normalized["department"].astype(str).str.strip().dropna())
        department_issues = sorted(value for value in observed_departments if value not in ALLOWED_DEPARTMENT_VALUES)
    quality = {
        "uploaded_rows": int(len(normalized)),
        "missing_required_columns": missing,
        "unexpected_columns": unexpected,
        "missing_value_counts": missing_counts,
        "salary_category_issues": salary_issues,
        "department_category_issues": department_issues,
        "left_present": "left" in normalized.columns,
    }
    errors = []
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
    if salary_issues:
        errors.append("Unexpected salary values: " + ", ".join(salary_issues))
    if department_issues:
        errors.append("Unexpected department values: " + ", ".join(department_issues))
    if missing_counts:
        errors.append("Required fields contain missing values.")
    return not errors, errors, quality


def build_review_queue(normalized: pd.DataFrame) -> pd.DataFrame:
    """Build a transparent review-priority queue with pandas only."""

    work = normalized[REQUIRED_FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["department"] = work["department"].astype(str).str.strip()
    work["salary"] = work["salary"].astype(str).str.strip().str.lower()
    if work[REQUIRED_FEATURE_COLUMNS].isna().any().any():
        bad_columns = work.columns[work.isna().any()].tolist()
        raise ValueError("Required feature values are missing or non-numeric: " + ", ".join(bad_columns))

    work.insert(0, "uploaded_row_id", range(1, len(work) + 1))
    work["project_intensity"] = work["average_monthly_hours"] / work["number_project"].clip(lower=1)

    scored_rows: list[dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        score = 0
        reasons: list[str] = []
        if row["average_monthly_hours"] >= 220:
            score += 20
            reasons.append("monthly hours at or above 220")
        if row["number_project"] >= 5:
            score += 15
            reasons.append("five or more projects")
        if row["tenure"] >= 4 and row["promotion_last_5years"] == 0:
            score += 15
            reasons.append("longer tenure without recent promotion")
        if row["last_evaluation"] >= 0.80 and row["number_project"] >= 5:
            score += 15
            reasons.append("high evaluation with heavy project load")
        if row["salary"] == "low" and row["average_monthly_hours"] >= 200:
            score += 15
            reasons.append("low salary with elevated hours")
        if row["satisfaction_level"] < 0.45:
            score += 20
            reasons.append("lower satisfaction signal")
        if row["project_intensity"] >= 45:
            score += 10
            reasons.append("high hours per project")

        review_score = min(score, 100)
        if review_score >= 70:
            band = "High"
        elif review_score >= 40:
            band = "Medium"
        else:
            band = "Low"
        row["review_score"] = int(review_score)
        row["review_band"] = band
        row["review_reasons"] = "; ".join(reasons) if reasons else "No elevated heuristic signal"
        row["scoring_mode"] = SCORING_MODE
        scored_rows.append(row)
    return pd.DataFrame(scored_rows).sort_values("review_score", ascending=False)


def _reason_counts(review_df: pd.DataFrame, *, high_only: bool = False, limit: int = 8) -> list[dict[str, Any]]:
    source = review_df[review_df["review_band"].eq("High")] if high_only else review_df
    counts: dict[str, int] = {}
    for reasons in source["review_reasons"].astype(str):
        for reason in reasons.split("; "):
            if reason and reason != "No elevated heuristic signal":
                counts[reason] = counts.get(reason, 0) + 1
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def _department_review_summary(review_df: pd.DataFrame) -> pd.DataFrame:
    reason_lookup: dict[str, str] = {}
    for department, group in review_df.groupby("department", dropna=False):
        reasons = _reason_counts(group, limit=1)
        reason_lookup[department] = reasons[0]["reason"] if reasons else "No elevated heuristic signal"
    summary = (
        review_df.assign(is_high=review_df["review_band"].eq("High"))
        .groupby("department", dropna=False)
        .agg(
            row_count=("uploaded_row_id", "count"),
            high_count=("is_high", "sum"),
            average_score=("review_score", "mean"),
        )
        .reset_index()
    )
    summary["high_rate"] = (summary["high_count"] / summary["row_count"]).round(3)
    summary["average_score"] = summary["average_score"].round(1)
    summary["top_reason"] = summary["department"].map(reason_lookup)
    return summary.sort_values(["high_count", "high_rate", "average_score"], ascending=[False, False, False])


def build_deterministic_insight_pack(review_df: pd.DataFrame) -> dict[str, Any]:
    """Build concrete deterministic insights before any optional AI briefing."""

    department_summary = _department_review_summary(review_df)
    high_departments = department_summary[department_summary["high_count"] > 0]
    rate_candidates = department_summary[department_summary["row_count"] >= 3].sort_values(
        ["high_rate", "high_count", "average_score"],
        ascending=[False, False, False],
    )
    top_reasons_overall = _reason_counts(review_df, high_only=False)
    top_reasons_high = _reason_counts(review_df, high_only=True)
    priority_rows = review_df.head(10)
    main_themes = [item["reason"] for item in top_reasons_high[:3]] or [item["reason"] for item in top_reasons_overall[:3]]

    return {
        "department_review_summary": department_summary.to_dict(orient="records"),
        "top_departments_by_high_count": high_departments.head(5).to_dict(orient="records"),
        "top_departments_by_high_rate": rate_candidates.head(5).to_dict(orient="records"),
        "top_review_reasons_overall": top_reasons_overall,
        "top_review_reasons_high_band": top_reasons_high,
        "top_priority_rows": priority_rows[
            [
                "uploaded_row_id",
                "review_score",
                "review_band",
                "department",
                "salary",
                "tenure",
                "number_project",
                "average_monthly_hours",
                "review_reasons",
            ]
        ].to_dict(orient="records"),
        "recommended_review_queue": {
            "first_priority_row_ids": priority_rows.head(5)["uploaded_row_id"].astype(int).tolist(),
            "departments_to_review": high_departments.head(5)["department"].tolist(),
            "main_review_themes": main_themes,
        },
        "largest_high_count_department": high_departments.head(1).to_dict(orient="records"),
        "highest_high_rate_department": rate_candidates.head(1).to_dict(orient="records"),
    }


def build_compact_openai_summary(
    review_df: pd.DataFrame,
    quality: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    """Return aggregate-only, identifier-free JSON for optional OpenAI briefing."""

    band_counts = review_df["review_band"].value_counts().to_dict()
    insights = build_deterministic_insight_pack(review_df)
    safe_notes = [
        "Identifier-like columns were excluded from the deterministic summary."
        if note.startswith("Identifier-like columns")
        else note
        for note in notes
    ]
    demo_data_note = ""
    if len(review_df) == 100 and set(review_df["department"]) == ALLOWED_DEPARTMENT_VALUES:
        demo_data_note = "This appears to be the built-in 100-row synthetic demo upload."
    elif len(review_df) < 30:
        demo_data_note = "This is a small upload; treat patterns as a lightweight demonstration."
    return {
        "row_count": int(quality.get("uploaded_rows", len(review_df))),
        "valid_row_count": int(len(review_df)),
        "invalid_row_count": int(max(quality.get("uploaded_rows", len(review_df)) - len(review_df), 0)),
        "scoring_mode": SCORING_MODE,
        "high_count": int(band_counts.get("High", 0)),
        "medium_count": int(band_counts.get("Medium", 0)),
        "low_count": int(band_counts.get("Low", 0)),
        "band_distribution": {
            "High": int(band_counts.get("High", 0)),
            "Medium": int(band_counts.get("Medium", 0)),
            "Low": int(band_counts.get("Low", 0)),
        },
        "department_review_summary": insights["department_review_summary"],
        "top_departments_by_high_count": insights["top_departments_by_high_count"],
        "top_departments_by_high_rate": insights["top_departments_by_high_rate"],
        "top_review_reasons_overall": insights["top_review_reasons_overall"],
        "top_review_reasons_high_band": insights["top_review_reasons_high_band"],
        "top_priority_rows": insights["top_priority_rows"],
        "recommended_review_queue": insights["recommended_review_queue"],
        "data_quality_notes": safe_notes,
        "responsible_use_boundary": RESPONSIBLE_USE_NOTE,
        "demo_data_note": demo_data_note,
    }


def _generate_ai_briefing(aggregate: dict[str, Any]) -> tuple[bool, str]:
    api_key = _config_value("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY is not configured for this Streamlit environment."

    model = _config_value("OPENAI_SUMMARY_MODEL", "gpt-5.4-mini")
    safe_payload = json.dumps(aggregate, indent=2, default=str)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Write a concrete HR analytics reviewer briefing from aggregate, anonymized data only. "
                        "Do not claim causal proof, do not say anyone will definitely leave, do not recommend "
                        "firing, discipline, or employment decisions, do not expose identifiers, and do not "
                        "describe the heuristic review score as a machine-learning probability."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create: 1) a 3 to 5 sentence executive summary, 2) rows to review first using "
                        "uploaded_row_id only, 3) departments to review first while distinguishing high_count "
                        "and high_rate, 4) main review drivers, 5) a data quality or sample-size caveat, "
                        "and 6) a responsible-use note. Use this compact aggregate JSON only:\n"
                        f"{safe_payload}"
                    ),
                },
            ],
        )
        text = getattr(response, "output_text", "") or ""
        return True, text.strip() or "The model returned an empty briefing."
    except Exception as exc:
        return False, str(exc)


def _render_quality_report(quality: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Uploaded rows", quality["uploaded_rows"])
    cols[1].metric("Missing required columns", len(quality["missing_required_columns"]))
    cols[2].metric("Unexpected columns", len(quality["unexpected_columns"]))
    cols[3].metric("Target `left` present", "Yes" if quality["left_present"] else "No")
    if quality["missing_required_columns"]:
        st.warning("Missing required columns: " + ", ".join(quality["missing_required_columns"]))
    if quality["missing_value_counts"]:
        st.warning("Missing value counts: " + json.dumps(quality["missing_value_counts"]))
    if quality["salary_category_issues"]:
        st.warning("Unexpected salary values: " + ", ".join(quality["salary_category_issues"]))
    if quality["department_category_issues"]:
        st.warning("Unexpected department values: " + ", ".join(quality["department_category_issues"]))
    if quality["unexpected_columns"]:
        st.caption("Unexpected non-PII columns were excluded: " + ", ".join(quality["unexpected_columns"]))


def _render_overview() -> None:
    st.subheader("What This Lab Shows")
    st.markdown(
        "The MLOps Mini-Lab is a local/dev extension that demonstrates how the original "
        "notebook-style workflow can be organized into reusable modules, CLI scripts, "
        "tracking, serving, orchestration, container configuration, and CI checks."
    )
    st.info(PUBLIC_REFERENCE_NOTE)
    st.markdown(
        "- **Package foundation:** reusable data prep, feature engineering, evaluation, training, and prediction helpers.\n"
        "- **Hosted CSV Insight:** Streamlit-only CSV upload, heuristic review scoring, and optional aggregate AI briefing.\n"
        "- **CLI pipeline:** prepare data, train candidates, evaluate the lab champion, and write local reports.\n"
        "- **MLflow tracking:** local experiment runs under ignored `mlruns/`.\n"
        "- **FastAPI serving:** optional local/dev service for the lab champion model.\n"
        "- **Docker Compose:** optional local stack for API, Streamlit, and MLflow UI.\n"
        "- **Airflow scaffold:** local/dev DAG contract that orchestrates the CLI scripts outside Streamlit.\n"
        "- **GitHub Actions CI:** compile checks, contract tests, static DAG validation, and Compose config validation."
    )
    st.caption(RESPONSIBLE_USE_NOTE)


def _render_pipeline_artifacts() -> None:
    st.subheader("Pipeline Artifact Status")
    st.caption("This section checks local lab outputs without requiring them to be committed.")
    rows = _file_status_rows()
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if any(row["Status"] == "Missing" for row in rows):
        st.info(
            "Local lab artifacts have not been generated in this environment. This is expected "
            "in hosted demos because generated models, reports, and MLflow runs are gitignored."
        )
        st.markdown("Run locally from the repo root:")
        _code_block("python scripts/mlops_run_pipeline.py")

    data_profile = _read_json(REPORTS_DIR / "data_profile.json")
    if data_profile:
        st.markdown("**Latest Data Profile**")
        split = data_profile.get("split", {})
        clean = data_profile.get("clean", {})
        cols = st.columns(4)
        cols[0].metric("Clean rows", clean.get("row_count", "N/A"))
        cols[1].metric("Duplicates removed", data_profile.get("duplicates_removed", "N/A"))
        cols[2].metric("Train rows", split.get("train_rows", "N/A"))
        cols[3].metric("Test rows", split.get("test_rows", "N/A"))

    model_card = _read_text(REPORTS_DIR / "model_card.md")
    if model_card:
        with st.expander("Preview local lab model card"):
            st.markdown(model_card)


def _render_training_mlflow() -> None:
    st.subheader("Training & MLflow")
    champion = _lab_champion()
    if champion:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Lab champion", str(champion.get("model_name", "N/A")))
        metric_cols[1].metric("Lab threshold", _format_number(champion.get("best_threshold"), 2))
        metric_cols[2].metric("Recall", _format_number(champion.get("best_recall")))
        metric_cols[3].metric("Precision", _format_number(champion.get("best_precision")))
        metric_cols[4].metric("F2", _format_number(champion.get("best_f2")))
        st.caption("The lab threshold may differ from the public app threshold. That is expected and local/dev only.")
    else:
        st.info(
            "No local lab champion report is available in this environment. This is expected "
            "unless you have run the MLOps pipeline locally."
        )
        st.markdown("Run locally from the repo root:")
        _code_block("python scripts/mlops_run_pipeline.py")

    _status_badge("MLflow run directory", (PROJECT_ROOT / "mlruns").exists())
    st.markdown("Run the local MLflow UI outside Streamlit:")
    _code_block("mlflow ui")
    st.markdown("Local-only MLflow URL, when started on your machine: http://localhost:5000")
    st.info("MLflow runs are gitignored and do not update `artifacts/v2/`.")


def _render_fastapi() -> None:
    st.subheader("FastAPI Serving")
    api_url = "http://127.0.0.1:8000"
    st.markdown(f"Configured local API URL: `{api_url}`")
    st.caption(
        "This page only checks read-only local API metadata endpoints after you click the button. "
        "It does not submit prediction payloads."
    )
    st.info(
        "`127.0.0.1` / `localhost` only works when FastAPI is running in the same local "
        "environment as Streamlit. In hosted Streamlit, it points to the hosted app container, "
        "not the viewer's laptop. Hosted CSV Insight does not require FastAPI."
    )

    if st.button("Check local API status"):
        health_ok, health_payload = _api_get(api_url, "/health")
        model_ok, model_payload = _api_get(api_url, "/model-info")
        if health_ok:
            st.success("/health is reachable")
            status_cols = st.columns(4)
            status_cols[0].metric("Artifact available", "Yes" if health_payload.get("model_artifact_available") else "No")
            status_cols[1].metric("Metadata available", "Yes" if health_payload.get("model_metadata_available") else "No")
            status_cols[2].metric("Loaded in memory", "Yes" if health_payload.get("model_loaded_in_memory") else "No")
            status_cols[3].metric("Ready for prediction", "Yes" if health_payload.get("model_ready") else "No")
            st.caption(
                "FastAPI uses lazy loading, so `Loaded in memory` can be No while artifacts are available "
                "and the model is ready to load on first prediction."
            )
            st.json(health_payload)
        else:
            st.info("Optional local API is not connected. This is expected unless you started FastAPI locally.")
            with st.expander("Technical connection details"):
                st.code(str(health_payload), language="text")
        if model_ok:
            st.success("/model-info is reachable")
            st.json(model_payload)
        else:
            st.info("Optional local API metadata is not connected. This is expected unless you started FastAPI locally.")
            with st.expander("Technical connection details"):
                st.code(str(model_payload), language="text")
    else:
        st.info("If the local API is offline, start it with one of these commands outside Streamlit.")

    _code_block("python -m uvicorn api.main:app --reload")
    _code_block("docker compose up api")

    with st.expander("Example /predict payload for manual local API testing"):
        st.json(
            {
                "satisfaction_level": 0.38,
                "last_evaluation": 0.86,
                "number_project": 5,
                "average_monthly_hours": 230,
                "tenure": 5,
                "work_accident": 0,
                "promotion_last_5years": 0,
                "department": "sales",
                "salary": "low",
            }
        )


def _render_online_csv_insight() -> None:
    st.subheader("Online CSV Insight")
    st.markdown(ONLINE_SANDBOX_NOTE)
    st.info(HEURISTIC_BOUNDARY_NOTE)
    template_cols = st.columns(2)
    with template_cols[0]:
        st.download_button(
            "Download minimal schema template",
            data=_minimal_template_csv().encode("utf-8"),
            file_name="salifort_csv_insight_schema_template.csv",
            mime="text/csv",
        )
    with template_cols[1]:
        st.download_button(
            "Download 100-row synthetic demo CSV",
            data=_synthetic_demo_csv().encode("utf-8"),
            file_name="salifort_csv_insight_100_row_demo.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader(
        "Upload a Salifort-style CSV",
        type=["csv"],
        help="Limit 2,000 rows. Files are read in memory and are not written to disk.",
    )
    if uploaded_file is None:
        st.caption("Required columns can use raw Salifort names or normalized app names.")
        return

    try:
        upload_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.warning("The uploaded CSV could not be read.")
        with st.expander("CSV read details"):
            st.code(str(exc), language="text")
        return

    if len(upload_df) > MAX_UPLOAD_ROWS:
        st.warning(f"Upload has {len(upload_df):,} rows. Please limit this sandbox to {MAX_UPLOAD_ROWS:,} rows.")
        return

    normalized, notes = normalize_uploaded_columns(upload_df)
    valid, errors, quality = _validate_upload_frame(normalized)
    _render_quality_report(quality)
    for note in notes:
        st.caption(note)
    if errors:
        st.warning("Resolve the validation issue before generating a review summary.")
        with st.expander("Validation details"):
            st.code("\n".join(errors), language="text")
        return

    try:
        review_df = build_review_queue(normalized)
    except Exception as exc:
        st.warning("The review queue could not be created from the uploaded CSV.")
        with st.expander("Review scoring details"):
            st.code(str(exc), language="text")
        return

    aggregate = build_compact_openai_summary(review_df, quality, notes)
    st.session_state["online_csv_insight_aggregate"] = aggregate

    cols = st.columns(5)
    cols[0].metric("Rows analyzed", aggregate["valid_row_count"])
    cols[1].metric("High", aggregate["high_count"])
    cols[2].metric("Medium", aggregate["medium_count"])
    cols[3].metric("Low", aggregate["low_count"])
    cols[4].metric("Scoring mode", SCORING_MODE)

    insights = build_deterministic_insight_pack(review_df)
    priority_ids = insights["recommended_review_queue"]["first_priority_row_ids"]
    largest_high = insights["largest_high_count_department"]
    highest_rate = insights["highest_high_rate_department"]
    top_reason = (insights["top_review_reasons_overall"] or [{"reason": "No elevated heuristic signal"}])[0]["reason"]
    top_high_reason = (insights["top_review_reasons_high_band"] or [{"reason": "No High-band rows"}])[0]["reason"]

    st.markdown("**Deterministic insight pack**")
    st.markdown(f"- Priority rows to review first: uploaded_row_id {', '.join(str(item) for item in priority_ids)}")
    if largest_high:
        item = largest_high[0]
        st.markdown(f"- Largest high-review count: {item['department']} with {int(item['high_count'])} High rows")
    else:
        st.markdown("- Largest high-review count: no High review-band rows in this upload")
    if highest_rate:
        item = highest_rate[0]
        st.markdown(
            f"- Highest high-review rate among departments with at least 3 rows: "
            f"{item['department']} at {item['high_rate']:.0%}"
        )
    else:
        st.markdown("- Highest high-review rate: no department has at least 3 rows")
    st.markdown(f"- Most common review reason overall: {top_reason}")
    st.markdown(f"- Most common review reason among High rows: {top_high_reason}")

    st.markdown("**Department review summary**")
    st.dataframe(pd.DataFrame(insights["department_review_summary"]), use_container_width=True, hide_index=True)

    st.markdown("**Top departments by high review count**")
    top_departments = pd.DataFrame(insights["top_departments_by_high_count"])
    if top_departments.empty:
        st.info("No departments have High review-band rows in this upload.")
    else:
        st.dataframe(top_departments, use_container_width=True, hide_index=True)
    st.markdown("**Top review reasons overall**")
    st.dataframe(pd.DataFrame(insights["top_review_reasons_overall"]), use_container_width=True, hide_index=True)
    st.markdown("**Top review reasons among High rows**")
    st.dataframe(pd.DataFrame(insights["top_review_reasons_high_band"]), use_container_width=True, hide_index=True)
    st.markdown("**Recommended first-pass review queue**")
    st.dataframe(review_df.head(5), use_container_width=True, hide_index=True)

    st.download_button(
        "Download review summary CSV",
        data=review_df[DOWNLOAD_COLUMNS].to_csv(index=False).encode("utf-8"),
        file_name="salifort_streamlit_review_summary.csv",
        mime="text/csv",
    )

    st.markdown("**Optional AI briefing**")
    st.caption("OpenAI receives only compact aggregate JSON, not raw CSV rows or identifier fields.")
    if not _config_value("OPENAI_API_KEY"):
        st.info("Add `OPENAI_API_KEY` in Streamlit secrets to enable the optional AI briefing.")
    if st.button("Generate AI briefing"):
        ok, briefing = _generate_ai_briefing(aggregate)
        if ok:
            st.markdown(briefing)
        else:
            st.info("AI briefing is not available right now.")
            with st.expander("Technical AI briefing details"):
                st.code(briefing, language="text")


def _render_docker() -> None:
    st.subheader("Docker Compose")
    st.markdown("Docker is an optional local/dev demo. It is not required for Streamlit Cloud.")
    st.markdown("Common local commands:")
    _code_block(
        "\n".join(
            [
                "docker compose build",
                "docker compose up",
                "docker compose up api",
                "docker compose up streamlit",
                "docker compose --profile mlflow up mlflow",
                "docker compose down",
            ]
        )
    )
    st.markdown(
        "- Local-only FastAPI docs: http://localhost:8000/docs\n"
        "- Local-only Streamlit: http://localhost:8501\n"
        "- Local-only MLflow UI: http://localhost:5000"
    )
    st.info(
        "These URLs are for local Docker Desktop usage only. They are not expected to work from "
        "the hosted Streamlit app. The API container mounts local `mlops/` artifacts; generated "
        "models are not baked into the image."
    )


def _render_airflow() -> None:
    st.subheader("Airflow DAG")
    st.markdown("DAG ID: `salifort_mlops_mini_lab_pipeline`")
    st.markdown("Task order:")
    st.code("prepare_data >> train_model >> evaluate_model >> validate_api_contract", language="text")
    st.markdown("Static validation command:")
    _code_block("python scripts/validate_mlops_airflow_dag.py")
    st.info(
        "No Airflow service is expected to be running in the hosted app. The DAG is included as "
        "a local/dev orchestration scaffold and validated statically. Streamlit does not trigger "
        "DAG runs, and the DAG does not write to public app artifacts."
    )


def _render_ci() -> None:
    st.subheader("CI & Validation")
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    _status_badge("GitHub Actions workflow", workflow_path.exists())
    if workflow_path.exists():
        st.markdown(
            "- `app-runtime-checks`: installs app requirements and compiles Streamlit runtime files.\n"
            "- `mlops-tests`: installs MLOps requirements, compiles lab/API/DAG files, validates the DAG, and runs contract tests.\n"
            "- `docker-config-check`: validates Docker Compose config and the optional MLflow profile."
        )
    st.markdown("Local validation commands:")
    _code_block(
        "\n".join(
            [
                "python -m pytest tests/test_data_contract.py tests/test_feature_pipeline.py tests/test_evaluation_metrics.py tests/test_training_pipeline.py tests/test_api_contract.py tests/test_prediction_service.py tests/test_airflow_dag_contract.py",
                "python scripts/validate_mlops_airflow_dag.py",
                "docker compose config",
                "docker compose --profile mlflow config",
            ]
        )
    )
    st.info(
        "CI validates code and configuration; it does not generate or publish lab model artifacts. "
        "It also does not deploy, publish images, install Airflow, or require generated lab artifacts."
    )


def _render_boundaries() -> None:
    st.subheader("Responsible Use & Boundaries")
    st.markdown(
        "- This is a portfolio demonstration, not production HR infrastructure.\n"
        "- Outputs support human review only and are not employment decisions.\n"
        "- The hosted CSV Insight score is a transparent heuristic, not a model probability.\n"
        "- The lab model and public app model are intentionally separate.\n"
        "- The public app remains weighted XGBoost at threshold `0.29`.\n"
        "- The original eight app pages remain artifact-backed and do not depend on the MLOps lab.\n"
        "- Streamlit does not run training, Docker, MLflow, Airflow, git, CI, shell commands, or background jobs."
    )
    st.caption(f"Last page render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render() -> None:
    st.title("MLOps Lab")
    st.caption(
        "Hosted CSV insight plus local/dev serving, tracking, orchestration, and CI extension."
    )
    st.warning(
        "Read-only page: this documents and inspects the MLOps Mini-Lab extension. "
        "It does not run training, trigger Airflow, start Docker containers, run MLflow, "
        "or replace the public artifact-backed model used by the main app."
    )
    st.info(HOSTED_DEMO_NOTE)

    tabs = st.tabs(
        [
            "Overview",
            "Online CSV Insight",
            "Pipeline Artifacts",
            "Training & MLflow",
            "FastAPI Serving",
            "Docker Compose",
            "Airflow DAG",
            "CI & Validation",
            "Responsible Use",
        ]
    )
    with tabs[0]:
        _render_overview()
    with tabs[1]:
        _render_online_csv_insight()
    with tabs[2]:
        _render_pipeline_artifacts()
    with tabs[3]:
        _render_training_mlflow()
    with tabs[4]:
        _render_fastapi()
    with tabs[5]:
        _render_docker()
    with tabs[6]:
        _render_airflow()
    with tabs[7]:
        _render_ci()
    with tabs[8]:
        _render_boundaries()
