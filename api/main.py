"""Optional FastAPI app for serving the Salifort MLOps Mini-Lab model."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.model_loader import (
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
    if state["model_ready"] and not state["model_loaded_in_memory"]:
        message = "Service is healthy. Model artifacts are available and the model will load on first prediction."
    elif state["model_ready"]:
        message = "Service is healthy. Model is loaded in memory and ready for prediction."
    else:
        message = "Service is healthy, but lab model artifacts are missing. Run python scripts/mlops_run_pipeline.py."
    return HealthResponse(
        status="ok",
        model_loaded=bool(state["model_loaded"]),
        model_artifact_available=bool(state["model_artifact_available"]),
        model_metadata_available=bool(state["model_metadata_available"]),
        model_loaded_in_memory=bool(state["model_loaded_in_memory"]),
        model_ready=bool(state["model_ready"]),
        message=message,
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


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
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
