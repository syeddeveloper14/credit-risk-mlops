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
