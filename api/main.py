"""Optional FastAPI app for serving the Salifort MLOps Mini-Lab model."""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, status

from api.model_loader import (
    MISSING_MODEL_MESSAGE,
    ModelUnavailableError,
    get_model_info,
    get_model_state,
    predict_batch,
    predict_one,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EmployeeFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

app = FastAPI(title="Salifort MLOps Mini-Lab API", version="0.1.0")


def require_prediction_token(authorization: str | None = Header(default=None)) -> None:
    """Optionally require a bearer token for prediction endpoints.

    Local/dev deployments stay open when SALIFORT_API_TOKEN is unset. Hosted
    deployments can set the env var to protect prediction requests without
    exposing the expected token in error responses.
    """

    expected_token = os.getenv("SALIFORT_API_TOKEN")
    if not expected_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Prediction endpoint requires a bearer token.",
        )
    provided_token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prediction endpoint token is invalid.",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = get_model_state()
    return HealthResponse(
        status="ok",
        model_loaded=bool(state["model_loaded"]),
        message="Service is healthy." if state["model_available"] else MISSING_MODEL_MESSAGE,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_model_info())


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_prediction_token)])
def predict(payload: EmployeeFeatures) -> PredictionResponse:
    try:
        return PredictionResponse(**predict_one(payload))
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(require_prediction_token)],
)
def batch_predict(payload: BatchPredictionRequest | list[EmployeeFeatures]) -> BatchPredictionResponse:
    records = payload if isinstance(payload, list) else payload.records
    try:
        predictions = [PredictionResponse(**item) for item in predict_batch(records)]
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    probabilities = [item.attrition_probability for item in predictions]
    return BatchPredictionResponse(
        row_count=len(predictions),
        high_risk_count=sum(1 for item in predictions if item.high_risk_flag),
        average_probability=sum(probabilities) / len(probabilities) if probabilities else 0.0,
        predictions=predictions,
    )
