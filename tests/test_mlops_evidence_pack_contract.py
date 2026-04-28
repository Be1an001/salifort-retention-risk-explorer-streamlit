from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "docs" / "demo-assets" / "mlops-evidence"
MLOPS_LAB_PAGE = REPO_ROOT / "app" / "pages" / "mlops_lab.py"
DEMO_GUIDE = REPO_ROOT / "docs" / "mlops-demo-guide.md"

EVIDENCE_FILES = [
    "README.md",
    "pipeline_run_summary.json",
    "training_evaluation_summary.json",
    "fastapi_health_example.json",
    "fastapi_model_info_example.json",
    "docker_compose_validation.md",
    "airflow_validation_summary.md",
    "github_actions_summary.md",
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_evidence_pack_files_exist() -> None:
    assert EVIDENCE_DIR.exists()
    for name in EVIDENCE_FILES:
        assert (EVIDENCE_DIR / name).exists(), name


def test_evidence_files_are_sanitized_and_lightweight() -> None:
    for path in EVIDENCE_DIR.iterdir():
        if not path.is_file():
            continue
        content = path.read_bytes()
        text = content.decode("utf-8")
        assert b"\x00" not in content
        assert len(content) < 100_000
        assert "C:\\Users\\" not in text
        assert "sk-" not in text


def test_json_evidence_contains_public_boundary() -> None:
    pipeline = json.loads(_text(EVIDENCE_DIR / "pipeline_run_summary.json"))
    training = json.loads(_text(EVIDENCE_DIR / "training_evaluation_summary.json"))
    fastapi_info = json.loads(_text(EVIDENCE_DIR / "fastapi_model_info_example.json"))

    combined = f"{pipeline} {training} {fastapi_info}"
    assert "weighted xgboost" in combined.lower()
    assert "0.29" in combined
    assert "not production" in combined.lower() or "not an employment decision" in combined.lower()


def test_mlops_lab_includes_evidence_tab_without_execution_calls() -> None:
    page = _text(MLOPS_LAB_PAGE)
    assert "MLOps Evidence" in page
    assert "docs/demo-assets/mlops-evidence" in page
    assert "subprocess.run" not in page
    assert "os.system" not in page
    assert "docker compose up --" not in page
    assert "airflow dags trigger" not in page


def test_demo_guide_covers_hosted_and_local_evidence_paths() -> None:
    guide = _text(DEMO_GUIDE)
    required_sections = [
        "Hosted Streamlit Demo",
        "Local MLOps Pipeline Demo",
        "Export Evidence Pack",
        "FastAPI Local Demo",
        "Docker Compose Demo",
        "MLflow Demo",
        "Airflow DAG Evidence",
        "GitHub Actions / CI",
        "What This Project Does Not Claim",
    ]
    for section in required_sections:
        assert section in guide
    assert "weighted xgboost" in guide.lower()
    assert "0.29" in guide


def test_no_generated_models_or_mlruns_are_staged() -> None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    staged = result.stdout.splitlines()
    assert not any(path.endswith(".joblib") for path in staged)
    assert not any(path.startswith("mlruns/") or path == "mlruns" for path in staged)
