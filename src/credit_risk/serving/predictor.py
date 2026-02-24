"""Turn a fitted pipeline into credit decisions: PD, score, decision and reason codes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import ScoringConfig
from ..data.schema import FEATURES, ID
from ..models.scorecard import ScorecardModel, ScoreScaling
from ..registry import ModelRegistry

REASON_TEXT: dict[str, str] = {
    "revolving_utilization": "Revolving balances are high relative to credit limits",
    "age": "Age profile of the applicant",
    "late_30_59": "Recent 30-59 day delinquencies",
    "late_60_89": "Recent 60-89 day delinquencies",
    "late_90": "Serious delinquency (90+ days late)",
    "total_past_due": "Number of past-due events",
    "any_90_late": "Serious delinquency on record",
    "debt_ratio": "Debt payments are high relative to income",
    "debt_ratio_clean": "Debt payments are high relative to income",
    "monthly_debt": "Total monthly debt payments",
    "monthly_income": "Income level",
    "income_per_person": "Income relative to household size",
    "income_missing": "Income not provided",
    "open_credit_lines": "Number of open credit lines",
    "lines_per_decade": "Number of credit lines relative to age",
    "real_estate_loans": "Number of real estate loans",
    "dependents": "Number of dependents",
    "past_due_code": "Unusual delinquency record",
    "utilization_over_1": "Revolving balances exceed credit limits",
}


class CreditRiskPredictor:
    def __init__(
        self,
        model: Pipeline,
        scoring: ScoringConfig | None = None,
        version: str = "dev",
        reference_goods: dict[str, float] | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.scoring = scoring or ScoringConfig()
        self.scaling = ScoreScaling(
            self.scoring.pdo, self.scoring.base_score, self.scoring.base_odds
        )
        self.version = version
        self.reference_goods = reference_goods or {}
        self.log_path = Path(log_path) if log_path else None

    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
        scoring: ScoringConfig,
        stage: str = "Production",
        log_path: str | Path | None = None,
    ) -> CreditRiskPredictor:
        mv = registry.latest(stage)
        if mv is None:
            raise LookupError(f"No model in stage {stage!r}; run the pipeline first")
        model = registry.load_model(mv.version)
