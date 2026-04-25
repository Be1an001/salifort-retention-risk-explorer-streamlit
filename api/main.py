"""Optional FastAPI app for serving the Salifort MLOps Mini-Lab model."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

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


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: EmployeeFeatures) -> PredictionResponse:
    try:
        return PredictionResponse(**predict_one(payload))
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch-predict", response_model=BatchPredictionResponse)
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
