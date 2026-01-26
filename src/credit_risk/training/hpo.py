"""Hyperparameter search with cross-validated ROC-AUC.

Uses Optuna (TPE sampler + median pruning across CV folds) when installed and falls back to
seeded random search otherwise. The default configuration is always evaluated first, so
tuning can only match or improve on it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..models.candidates import DEFAULT_PARAMS, SEARCH_SPACES, build_model

log = logging.getLogger(__name__)


@dataclass
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    best_score: float
    backend: str
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


def _sample_random(space: dict[str, tuple], rng: np.random.Generator) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "float":
            lo, hi = spec[1], spec[2]
            log_scale = len(spec) > 3 and spec[3] == "log"
            params[name] = float(
                np.exp(rng.uniform(np.log(lo), np.log(hi))) if log_scale else rng.uniform(lo, hi)
            )
        elif kind == "cat":
            params[name] = spec[1][int(rng.integers(len(spec[1])))]
        else:
            raise ValueError(f"Unknown parameter kind {kind!r}")
    return params


def _suggest(trial, space: dict[str, tuple]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, spec[1], spec[2], log=len(spec) > 3 and spec[3] == "log"
            )
        else:
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def cv_auc(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
