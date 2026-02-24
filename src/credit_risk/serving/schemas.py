from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicantIn(BaseModel):
    applicant_id: int | None = None
    revolving_utilization: float = Field(..., ge=0)
    age: int = Field(..., ge=18, le=120)
    late_30_59: int = Field(..., ge=0)
    debt_ratio: float = Field(..., ge=0)
    monthly_income: float | None = Field(None, ge=0)
    open_credit_lines: int = Field(..., ge=0)
    late_90: int = Field(..., ge=0)
    real_estate_loans: int = Field(..., ge=0)
    late_60_89: int = Field(..., ge=0)
    dependents: int | None = Field(None, ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "applicant_id": 1001,
                    "revolving_utilization": 0.77,
                    "age": 45,
                    "late_30_59": 2,
                    "debt_ratio": 0.80,
                    "monthly_income": 9120,
                    "open_credit_lines": 13,
                    "late_90": 0,
                    "real_estate_loans": 6,
                    "late_60_89": 0,
                    "dependents": 2,
                }
            ]
        }
    }


class Reason(BaseModel):
    code: str
    text: str


class PredictionOut(BaseModel):
    applicant_id: int | None
    probability_of_default: float
    credit_score: int
    decision: str
    reasons: list[Reason]
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    model_type: str
    stage: str
    metrics: dict[str, float]
