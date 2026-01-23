"""Candidate training and model selection.

For each candidate family: tune on the training split with cross-validation, refit on the full
training split, score the validation split. The best family by the selection metric is refit
on train + validation; the test split stays untouched until evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import Config
from ..data.schema import FEATURES, TARGET
from ..evaluation import classification_metrics
from ..models import build_model
from ..tracking import Tracker
from .hpo import tune

log = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    params: dict[str, Any]
    cv_auc: float
    valid_metrics: dict[str, float]
    backend: str
    seconds: float
    run_id: str | None = None


@dataclass
class TrainingOutcome:
    best: CandidateResult
    candidates: list[CandidateResult] = field(default_factory=list)
    model: Pipeline | None = None

    def leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model": c.name,
                    "cv_auc": c.cv_auc,
                    **{
                        f"valid_{k}": v
                        for k, v in c.valid_metrics.items()
                        if k in ("roc_auc", "ks", "brier", "ece")
                    },
                    "hpo": c.backend,
                    "seconds": c.seconds,
                }
