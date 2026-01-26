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
    report: Callable[[int, float], None] | None = None,
) -> float:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for i, (tr, va) in enumerate(splitter.split(X, y)):
        model = build_model(name, params, seed).fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
        if report is not None:
            report(i, float(np.mean(scores)))
    return float(np.mean(scores))


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 25,
    cv_folds: int = 3,
    seed: int = 42,
    timeout_s: int | None = None,
    backend: str = "auto",
) -> TuningResult:
    space = SEARCH_SPACES[name]
    defaults = DEFAULT_PARAMS[name]
    if n_trials <= 1 or not space:
        score = cv_auc(name, defaults, X, y, cv_folds, seed)
        return TuningResult(name, dict(defaults), score, "defaults")

    use_optuna = backend in ("auto", "optuna")
    if use_optuna:
        try:
            import optuna
        except ImportError:
            if backend == "optuna":
                raise
            use_optuna = False

    if use_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.enqueue_trial(defaults)

        def objective(trial) -> float:
            params = _suggest(trial, space)

            def report(step: int, value: float) -> None:
                trial.report(value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return cv_auc(name, params, X, y, cv_folds, seed, report)

        study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
        history = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        return TuningResult(
            name, dict(study.best_params), float(study.best_value), "optuna", history
        )

    rng = np.random.default_rng(seed)
    rows, best_params, best_score = [], dict(defaults), -np.inf
    start = time.monotonic()
    for i in range(n_trials):
        params = dict(defaults) if i == 0 else _sample_random(space, rng)
        score = cv_auc(name, params, X, y, cv_folds, seed)
        rows.append({"number": i, "value": score, **{f"params_{k}": v for k, v in params.items()}})
        if score > best_score:
            best_params, best_score = params, score
        if timeout_s is not None and time.monotonic() - start > timeout_s:
            log.info("HPO timeout after %d trials", i + 1)
            break
    return TuningResult(name, best_params, float(best_score), "random", pd.DataFrame(rows))
