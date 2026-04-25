from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import api.model_loader as model_loader
from api.main import app


client = TestClient(app)


def _payload() -> dict[str, object]:
    return {
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


def test_predict_returns_controlled_503_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_loader, "_MODEL_CACHE", None)
    monkeypatch.setattr(model_loader, "_MODEL_PATH_CACHE", None)
    monkeypatch.setattr(model_loader, "LAB_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(model_loader, "LAB_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(model_loader, "STABLE_CHAMPION_MODEL_PATH", tmp_path / "models" / "champion_model.joblib")
    monkeypatch.setattr(model_loader, "LEGACY_LAB_CHAMPION_MODEL_PATH", tmp_path / "models" / "lab_champion.joblib")
    monkeypatch.setattr(model_loader, "EVALUATION_SUMMARY_PATH", tmp_path / "reports" / "evaluation_summary.json")
    monkeypatch.setattr(model_loader, "TRAINING_RESULTS_JSON_PATH", tmp_path / "reports" / "training_results.json")

    response = client.post("/predict", json=_payload())
    assert response.status_code == 503
    assert "mlops_run_pipeline.py" in response.json()["detail"]


def test_predict_returns_required_fields_when_model_exists() -> None:
    if not model_loader.is_model_available():
        pytest.skip("Run python scripts/mlops_run_pipeline.py to export a lab model.")

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    payload = response.json()
    assert set(
        [
            "attrition_probability",
            "threshold",
            "high_risk_flag",
            "review_band",
            "model_name",
            "model_scope",
            "responsible_use_note",
        ]
    ).issubset(payload)
    assert payload["model_scope"] == "mlops-mini-lab"
    assert "not an employment decision" in payload["responsible_use_note"]


def test_batch_predict_returns_row_count_and_predictions_when_model_exists() -> None:
    if not model_loader.is_model_available():
        pytest.skip("Run python scripts/mlops_run_pipeline.py to export a lab model.")

    response = client.post("/batch-predict", json={"records": [_payload(), _payload()]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 2
    assert len(payload["predictions"]) == 2
    assert "responsible_use_note" in payload
