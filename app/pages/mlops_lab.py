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
    "The MLOps Mini-Lab and external API mode do not replace that public model truth."
)
RESPONSIBLE_USE_NOTE = (
    "Portfolio demonstration and human review support only. This page is not an employment decision system."
)
HOSTED_DEMO_NOTE = (
    "Hosted demo note: local FastAPI, MLflow, Docker Compose, Airflow, and generated lab model "
    "artifacts are local/dev components. They are not expected to be running inside the hosted "
    "Streamlit app. External API scoring works only when `SALIFORT_API_URL` points to a "
    "separately deployed backend. This page is a read-only overview and status inspector."
)
EXTERNAL_MODE_NOTE = (
    "External FastAPI mode lets Streamlit Cloud call a separately deployed MLOps API. "
    "Uploaded CSVs are kept in memory, normalized to Salifort feature columns, stripped "
    "of identifier-like fields, and sent only to the configured `/batch-predict` endpoint."
)

RAW_RENAME_MAP = {
    "average_montly_hours": "average_monthly_hours",
    "time_spend_company": "tenure",
    "Work_accident": "work_accident",
    "Department": "department",
}
API_FEATURE_COLUMNS = [
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
PII_COLUMN_HINTS = (
    "name",
    "email",
    "employee",
    "emp_id",
    "id",
    "phone",
    "address",
    "street",
)
MAX_UPLOAD_ROWS = 2_000


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


def _api_json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str = "",
    timeout: int = 12,
) -> tuple[bool, dict[str, Any] | str]:
    """Call a JSON API endpoint without exposing connection details by default."""

    url = base_url.rstrip("/") + path
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return True, json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {exc.reason}. {detail}".strip()
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
    return any(hint in normalized for hint in PII_COLUMN_HINTS)


def _prepare_external_api_records(upload_df: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame, list[str]]:
    """Normalize uploaded rows and return API-safe records.

    The target column and unexpected columns are intentionally excluded from the
    API payload. Identifier-like columns are reported as removed, but never sent.
    """

    if upload_df.empty:
        raise ValueError("Uploaded CSV does not contain any rows.")
    if len(upload_df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Upload has {len(upload_df):,} rows. Please limit external scoring to {MAX_UPLOAD_ROWS:,} rows.")

    normalized = upload_df.rename(columns=RAW_RENAME_MAP).copy()
    missing = [column for column in API_FEATURE_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError("Missing required Salifort feature columns: " + ", ".join(missing))

    notes: list[str] = []
    pii_like = [column for column in normalized.columns if _is_pii_like_column(column)]
    if pii_like:
        notes.append("Identifier-like columns excluded from API payload: " + ", ".join(sorted(pii_like)))
    if "left" in normalized.columns:
        notes.append("Target column `left` was excluded from API payload.")
    unexpected = [
        column
        for column in normalized.columns
        if column not in set(API_FEATURE_COLUMNS + ["left"]) and column not in pii_like
    ]
    if unexpected:
        notes.append("Unexpected columns excluded from API payload: " + ", ".join(sorted(unexpected)))

    payload_df = normalized[API_FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        payload_df[column] = pd.to_numeric(payload_df[column], errors="coerce")
    payload_df["department"] = payload_df["department"].astype(str).str.strip()
    payload_df["salary"] = payload_df["salary"].astype(str).str.strip().str.lower()

    if payload_df[API_FEATURE_COLUMNS].isna().any().any():
        bad_columns = payload_df.columns[payload_df.isna().any()].tolist()
        raise ValueError("Some required feature values are missing or non-numeric: " + ", ".join(bad_columns))

    records = json.loads(payload_df.to_json(orient="records"))
    return records, payload_df, notes or ["No extra columns were included in the API payload."]


def _build_external_scoring_summary(
    payload_df: pd.DataFrame,
    api_response: dict[str, Any],
    data_quality_notes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions = api_response.get("predictions", [])
    if len(predictions) != len(payload_df):
        raise ValueError("API response row count did not match the uploaded valid row count.")

    rows: list[dict[str, Any]] = []
    for index, (record, prediction) in enumerate(zip(payload_df.to_dict(orient="records"), predictions), start=1):
        rows.append(
            {
                "row_number": index,
                "department": record["department"],
                "salary": record["salary"],
                "attrition_probability": float(prediction.get("attrition_probability", 0.0)),
                "review_band": prediction.get("review_band", "unknown"),
                "high_risk_flag": bool(prediction.get("high_risk_flag", False)),
                "threshold": prediction.get("threshold"),
            }
        )
    results_df = pd.DataFrame(rows).sort_values("attrition_probability", ascending=False)
    department_summary = (
        results_df.groupby("department", dropna=False)
        .agg(
            row_count=("row_number", "count"),
            high_risk_count=("high_risk_flag", "sum"),
            average_probability=("attrition_probability", "mean"),
        )
        .reset_index()
        .sort_values(["high_risk_count", "average_probability"], ascending=[False, False])
    )
    band_counts = results_df["review_band"].value_counts().to_dict()
    aggregate = {
        "row_count": int(api_response.get("row_count", len(results_df))),
        "valid_row_count": int(len(results_df)),
        "scoring_mode": "external_fastapi",
        "high_count": int(band_counts.get("high", 0)),
        "medium_count": int(band_counts.get("medium", 0)),
        "low_count": int(band_counts.get("low", 0)),
        "average_probability": float(api_response.get("average_probability", results_df["attrition_probability"].mean())),
        "top_departments": department_summary.head(5).to_dict(orient="records"),
        "top_review_rows": results_df.head(10)[
            ["row_number", "department", "salary", "attrition_probability", "review_band"]
        ].to_dict(orient="records"),
        "data_quality_notes": data_quality_notes,
        "responsible_use_boundary": RESPONSIBLE_USE_NOTE,
    }
    return results_df, department_summary, aggregate


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
                        "Write a concise HR analytics reviewer briefing from aggregate, anonymized data only. "
                        "Do not claim causal proof, do not say anyone will definitely leave, do not recommend "
                        "firing, and preserve the responsible-use boundary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create a 3 to 5 sentence summary, 3 review-focus bullets, a data quality caveat "
                        "if needed, and a responsible-use note from this compact aggregate JSON:\n"
                        f"{safe_payload}"
                    ),
                },
            ],
        )
        text = getattr(response, "output_text", "") or ""
        return True, text.strip() or "The model returned an empty briefing."
    except Exception as exc:
        return False, str(exc)


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
        "- **CLI pipeline:** prepare data, train candidates, evaluate the lab champion, and write local reports.\n"
        "- **MLflow tracking:** local experiment runs under ignored `mlruns/`.\n"
        "- **FastAPI serving:** optional local or separately deployed service for the lab champion model.\n"
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
    api_url = os.getenv("SALIFORT_API_URL", "http://127.0.0.1:8000")
    st.markdown(f"Configured local API URL: `{api_url}`")
    st.caption(
        "This page only checks read-only API metadata endpoints after you click the button. "
        "It does not submit prediction payloads."
    )
    st.info(
        "`127.0.0.1` / `localhost` only works when FastAPI is running in the same local "
        "environment as Streamlit. In hosted Streamlit, it points to the hosted app container, "
        "not the viewer's laptop."
    )

    if st.button("Check local API status"):
        health_ok, health_payload = _api_get(api_url, "/health")
        model_ok, model_payload = _api_get(api_url, "/model-info")
        if health_ok:
            st.success("/health is reachable")
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
        st.info(
            "If the API is offline, start it locally with one of these commands outside Streamlit."
        )

    _code_block("python -m uvicorn api.main:app --reload")
    _code_block("docker compose up api")

    with st.expander("Example /predict payload for manual API testing"):
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


def _render_online_api_scoring() -> None:
    st.subheader("Online API Scoring")
    st.markdown(EXTERNAL_MODE_NOTE)

    external_api_url = _config_value("SALIFORT_API_URL")
    external_api_token = _config_value("SALIFORT_API_TOKEN")
    if not external_api_url:
        st.info(
            "No external FastAPI endpoint is configured. Deploy the FastAPI service separately "
            "and add `SALIFORT_API_URL` in Streamlit secrets."
        )
        return

    st.markdown(f"Configured external endpoint: `{external_api_url.rstrip('/')}`")
    if external_api_token:
        st.caption("Bearer-token authentication is configured for scoring requests.")
    else:
        st.caption("No API token is configured. This is acceptable for open demos but not recommended for shared services.")

    if st.button("Check external API status"):
        health_ok, health_payload = _api_json_request(external_api_url, "/health")
        model_ok, model_payload = _api_json_request(external_api_url, "/model-info")
        if health_ok:
            st.success("External /health is reachable")
            st.json(health_payload)
        else:
            st.info("External API status is not available right now.")
            with st.expander("Technical API error details"):
                st.code(str(health_payload), language="text")
        if model_ok:
            st.success("External /model-info is reachable")
            st.json(model_payload)
        else:
            st.info("External API model metadata is not available right now.")
            with st.expander("Technical API error details"):
                st.code(str(model_payload), language="text")

    uploaded_file = st.file_uploader(
        "Upload Salifort CSV for external API scoring",
        type=["csv"],
        help="Limit 2,000 rows. Files are read in memory and are not written to disk.",
    )
    if uploaded_file is None:
        st.caption("Required columns can use raw Salifort names or normalized app names.")
        return

    try:
        upload_df = pd.read_csv(uploaded_file)
        records, payload_df, notes = _prepare_external_api_records(upload_df)
    except Exception as exc:
        st.warning("The uploaded CSV could not be prepared for external scoring.")
        with st.expander("CSV validation details"):
            st.code(str(exc), language="text")
        return

    st.success(f"Prepared {len(records):,} feature records for external API scoring.")
    for note in notes:
        st.caption(note)

    if st.button("Run external API batch scoring"):
        ok, response_payload = _api_json_request(
            external_api_url,
            "/batch-predict",
            method="POST",
            payload={"records": records},
            token=external_api_token,
            timeout=30,
        )
        if not ok:
            st.warning("External API scoring is not available right now.")
            with st.expander("Technical API error details"):
                st.code(str(response_payload), language="text")
            return
        try:
            results_df, department_summary, aggregate = _build_external_scoring_summary(
                payload_df,
                response_payload if isinstance(response_payload, dict) else {},
                notes,
            )
        except Exception as exc:
            st.warning("External API returned an unexpected scoring response.")
            with st.expander("Technical API error details"):
                st.code(str(exc), language="text")
            return

        st.session_state["external_api_scoring_results"] = results_df
        st.session_state["external_api_department_summary"] = department_summary
        st.session_state["external_api_aggregate"] = aggregate

    results_df = st.session_state.get("external_api_scoring_results")
    department_summary = st.session_state.get("external_api_department_summary")
    aggregate = st.session_state.get("external_api_aggregate")
    if results_df is None or aggregate is None:
        return

    cols = st.columns(4)
    cols[0].metric("Scored rows", aggregate["valid_row_count"])
    cols[1].metric("High review band", aggregate["high_count"])
    cols[2].metric("Average probability", _format_number(aggregate["average_probability"]))
    cols[3].metric("Scoring mode", aggregate["scoring_mode"])

    st.markdown("**Top review rows**")
    st.dataframe(results_df.head(20), use_container_width=True, hide_index=True)

    if department_summary is not None and not department_summary.empty:
        st.markdown("**Department summary**")
        st.dataframe(department_summary, use_container_width=True, hide_index=True)

    st.download_button(
        "Download review summary CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="salifort_external_api_review_summary.csv",
        mime="text/csv",
    )

    st.markdown("**Optional AI briefing**")
    st.caption("OpenAI receives only compact aggregate JSON, not raw CSV rows or identifier fields.")
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
        "- The lab model and public app model are intentionally separate.\n"
        "- The public app remains weighted XGBoost at threshold `0.29`.\n"
        "- The original eight app pages remain artifact-backed and do not depend on the MLOps lab.\n"
        "- Streamlit does not run training, Docker, MLflow, Airflow, git, CI, or background jobs."
    )
    st.caption(f"Last page render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render() -> None:
    st.title("MLOps Lab")
    st.caption(
        "Local/dev serving, tracking, orchestration, and CI extension for the Salifort portfolio app."
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
            "Pipeline Artifacts",
            "Training & MLflow",
            "FastAPI Serving",
            "Online API Scoring",
            "Docker Compose",
            "Airflow DAG",
            "CI & Validation",
            "Responsible Use",
        ]
    )
    with tabs[0]:
        _render_overview()
    with tabs[1]:
        _render_pipeline_artifacts()
    with tabs[2]:
        _render_training_mlflow()
    with tabs[3]:
        _render_fastapi()
    with tabs[4]:
        _render_online_api_scoring()
    with tabs[5]:
        _render_docker()
    with tabs[6]:
        _render_airflow()
    with tabs[7]:
        _render_ci()
    with tabs[8]:
        _render_boundaries()
