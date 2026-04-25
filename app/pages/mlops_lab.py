from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLOPS_ROOT = PROJECT_ROOT / "mlops"
REPORTS_DIR = MLOPS_ROOT / "reports"
MODELS_DIR = MLOPS_ROOT / "models"
PROCESSED_DIR = MLOPS_ROOT / "data" / "processed"

PUBLIC_REFERENCE_NOTE = (
    "The public app remains artifact-backed with weighted XGBoost at threshold 0.29. "
    "The MLOps Mini-Lab is local/dev only and does not replace that public model truth."
)
RESPONSIBLE_USE_NOTE = (
    "Portfolio demonstration and human review support only. This page is not an employment decision system."
)
HOSTED_DEMO_NOTE = (
    "Hosted demo note: FastAPI, MLflow, Docker Compose, Airflow, and generated lab model "
    "artifacts are local/dev components. They are not expected to be running in the hosted "
    "Streamlit app. This page is a read-only overview and status inspector."
)


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
        "- **FastAPI serving:** optional local service for the lab champion model.\n"
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
        _render_docker()
    with tabs[5]:
        _render_airflow()
    with tabs[6]:
        _render_ci()
    with tabs[7]:
        _render_boundaries()
