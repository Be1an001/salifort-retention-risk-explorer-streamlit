from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def test_batch_predict_remains_open_when_token_is_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("SALIFORT_API_TOKEN", raising=False)

    response = client.post("/batch-predict", json={"records": [_payload()]})

    assert response.status_code not in {401, 403}


def test_batch_predict_requires_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("SALIFORT_API_TOKEN", "expected-token")

    response = client.post("/batch-predict", json={"records": [_payload()]})

    assert response.status_code == 401
    assert "requires a bearer token" in response.json()["detail"]


def test_batch_predict_rejects_wrong_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("SALIFORT_API_TOKEN", "expected-token")

    response = client.post(
        "/batch-predict",
        json={"records": [_payload()]},
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 403
    assert "invalid" in response.json()["detail"]


def test_batch_predict_accepts_correct_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("SALIFORT_API_TOKEN", "expected-token")

    response = client.post(
        "/batch-predict",
        json={"records": [_payload()]},
        headers={"Authorization": "Bearer expected-token"},
    )

    assert response.status_code not in {401, 403}
