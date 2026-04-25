from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.main import app
from api.schemas import EmployeeFeatures


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


def test_api_app_imports_and_health_responds() -> None:
    assert app.title == "Salifort MLOps Mini-Lab API"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_model_info_responds_even_without_asserting_model_presence() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "salifort-mlops-mini-lab-api"
    assert payload["model_available"] in {True, False}
    assert "public_reference_note" in payload


def test_employee_features_accepts_normalized_payload() -> None:
    features = EmployeeFeatures.model_validate(_payload())
    assert features.average_monthly_hours == 230
    assert features.department == "sales"


def test_employee_features_accepts_raw_alias_payload() -> None:
    raw_payload = _payload()
    raw_payload["average_montly_hours"] = raw_payload.pop("average_monthly_hours")
    raw_payload["time_spend_company"] = raw_payload.pop("tenure")
    raw_payload["Work_accident"] = raw_payload.pop("work_accident")
    raw_payload["Department"] = raw_payload.pop("department")
    features = EmployeeFeatures.model_validate(raw_payload)
    assert features.average_monthly_hours == 230
    assert features.tenure == 5
    assert features.work_accident == 0
    assert features.department == "sales"


@pytest.mark.parametrize(
    ("field", "value"),
    [("salary", "very_high"), ("department", "unknown")],
)
def test_invalid_salary_or_department_is_rejected(field: str, value: str) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate(payload)
