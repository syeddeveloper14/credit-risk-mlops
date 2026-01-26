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
                for c in self.candidates
            ]
        ).sort_values("valid_roc_auc", ascending=False)


def train_and_select(
    cfg: Config, train: pd.DataFrame, valid: pd.DataFrame, tracker: Tracker
) -> TrainingOutcome:
    X_train, y_train = train[FEATURES], train[TARGET]
    X_valid, y_valid = valid[FEATURES], valid[TARGET]
    results: list[CandidateResult] = []

    for name in cfg.models.candidates:
        start = time.monotonic()
        with tracker.start_run(f"candidate-{name}", tags={"stage": "train", "model": name}):
            tuning = tune(
                name,
                X_train,
                y_train,
                n_trials=cfg.hpo.n_trials,
                cv_folds=cfg.hpo.cv_folds,
                seed=cfg.hpo.seed,
                timeout_s=cfg.hpo.timeout_s,
            )
            model = build_model(name, tuning.best_params, cfg.seed).fit(X_train, y_train)
            metrics = classification_metrics(y_valid, model.predict_proba(X_valid)[:, 1])
            tracker.log_params({"model": name, "hpo_backend": tuning.backend, **tuning.best_params})
            tracker.log_metrics(
                {"cv_auc": tuning.best_score, **{f"valid_{k}": v for k, v in metrics.items()}}
            )
            run_id = tracker.run_id
        seconds = time.monotonic() - start
        log.info(
            "%-10s cv_auc=%.4f valid_auc=%.4f (%.1fs)",
            name,
            tuning.best_score,
            metrics["roc_auc"],
            seconds,
        )
        results.append(
            CandidateResult(
                name,
                tuning.best_params,
                tuning.best_score,
                metrics,
                tuning.backend,
                seconds,
                run_id,
            )
        )

    key = cfg.models.selection_metric
    best = max(results, key=lambda r: r.valid_metrics[key])
    full = pd.concat([train, valid])
    with tracker.start_run(f"final-{best.name}", tags={"stage": "final", "model": best.name}):
        model = build_model(best.name, best.params, cfg.seed).fit(full[FEATURES], full[TARGET])
        tracker.log_params({"model": best.name, **best.params, "train_rows": len(full)})
        tracker.log_metrics({f"selected_valid_{key}": best.valid_metrics[key]})
    return TrainingOutcome(best=best, candidates=results, model=model)
