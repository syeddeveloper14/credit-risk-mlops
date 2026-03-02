"""FastAPI service for the Production model.

``CREDIT_RISK_CONFIG`` points at the pipeline config (default ``configs/pipeline.yaml``);
the service loads whatever version is in the Production stage of the registry.
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI

from ..config import Config, load_config
from ..registry import ModelRegistry
from .predictor import REASON_TEXT, CreditRiskPredictor
from .schemas import ApplicantIn, ModelInfo, PredictionOut, Reason


def _to_out(row: pd.Series) -> PredictionOut:
    return PredictionOut(
        applicant_id=None if pd.isna(row.get("applicant_id")) else int(row["applicant_id"]),
        probability_of_default=float(row["probability_of_default"]),
        credit_score=int(row["credit_score"]),
        decision=str(row["decision"]),
        reasons=[Reason(code=r, text=REASON_TEXT.get(r, r)) for r in row["reasons"]],
        model_version=str(row["model_version"]),
    )


def create_app(predictor: CreditRiskPredictor | None = None, cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("CREDIT_RISK_CONFIG", "configs/pipeline.yaml"))
    registry = ModelRegistry(cfg.registry.root, cfg.registry.model_name)
    if predictor is None:
        predictor = CreditRiskPredictor.from_registry(
            registry, cfg.scoring, log_path=cfg.monitoring.prediction_log
        )
    app = FastAPI(title="Credit Risk API", version=predictor.version)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": predictor.version}

    @app.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        mv = registry.latest("Production")
        return ModelInfo(
