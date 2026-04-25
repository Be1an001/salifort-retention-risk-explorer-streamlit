"""Pydantic schemas for the optional Salifort MLOps Mini-Lab API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from salifort_mlops.config import ALLOWED_DEPARTMENT_VALUES, ALLOWED_SALARY_VALUES

DepartmentValue = Literal[
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
]
SalaryValue = Literal["low", "medium", "high"]

RESPONSIBLE_USE_NOTE = (
    "This prediction is for portfolio demonstration and human review support only. "
    "It is not an employment decision."
)
PUBLIC_REFERENCE_NOTE = (
    "The public Streamlit app remains artifact-backed with weighted XGBoost at "
    "threshold 0.29; this API serves only the local/dev MLOps lab model."
)


class EmployeeFeatures(BaseModel):
    """Normalized employee feature payload with selected raw CSV aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    satisfaction_level: float = Field(ge=0, le=1)
    last_evaluation: float = Field(ge=0, le=1)
    number_project: int = Field(ge=0)
    average_monthly_hours: float = Field(
        ge=0,
        validation_alias=AliasChoices("average_monthly_hours", "average_montly_hours"),
    )
    tenure: float = Field(ge=0, validation_alias=AliasChoices("tenure", "time_spend_company"))
    work_accident: int = Field(
        ge=0,
        le=1,
        validation_alias=AliasChoices("work_accident", "Work_accident"),
    )
    promotion_last_5years: int = Field(ge=0, le=1)
    department: DepartmentValue = Field(
        validation_alias=AliasChoices("department", "Department"),
        description=f"One of: {', '.join(ALLOWED_DEPARTMENT_VALUES)}",
    )
    salary: SalaryValue = Field(description=f"One of: {', '.join(ALLOWED_SALARY_VALUES)}")


class PredictionResponse(BaseModel):
    attrition_probability: float
    threshold: float
    high_risk_flag: bool
    review_band: Literal["low", "medium", "high"]
    model_name: str
    model_version: str
    model_scope: Literal["mlops-mini-lab"] = "mlops-mini-lab"
    public_reference_note: str = PUBLIC_REFERENCE_NOTE
    responsible_use_note: str = RESPONSIBLE_USE_NOTE


class BatchPredictionRequest(BaseModel):
    records: list[EmployeeFeatures]


class BatchPredictionResponse(BaseModel):
    row_count: int
    high_risk_count: int
    average_probability: float
    predictions: list[PredictionResponse]
    responsible_use_note: str = RESPONSIBLE_USE_NOTE


class ModelInfoResponse(BaseModel):
    service: str
    model_available: bool
    model_name: str | None = None
    model_version: str | None = None
    threshold: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str | None] = Field(default_factory=dict)
    message: str
    public_reference_note: str = PUBLIC_REFERENCE_NOTE
    responsible_use_note: str = RESPONSIBLE_USE_NOTE


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str
