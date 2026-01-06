"""Candidate model families, their defaults and hyperparameter search spaces."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import make_preprocessor
from .scorecard import ScorecardModel

# (kind, low, high[, "log"]) for numeric parameters; ("cat", [choices]) for categorical ones.
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "scorecard": {
        "max_bins": ("int", 4, 12),
        "min_bin_share": ("float", 0.02, 0.10),
        "C": ("float", 0.01, 10.0, "log"),
    },
    "logistic": {"C": ("float", 1e-3, 10.0, "log")},
    "hgb": {
        "learning_rate": ("float", 0.02, 0.2, "log"),
        "max_iter": ("int", 100, 500),
        "max_leaf_nodes": ("int", 8, 64),
        "min_samples_leaf": ("int", 20, 300),
        "l2_regularization": ("float", 1e-3, 10.0, "log"),
    },
    "lightgbm": {
        "learning_rate": ("float", 0.02, 0.2, "log"),
        "n_estimators": ("int", 100, 600),
        "num_leaves": ("int", 8, 64),
        "min_child_samples": ("int", 20, 300),
        "reg_lambda": ("float", 1e-3, 10.0, "log"),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "scorecard": {"max_bins": 8, "min_bin_share": 0.05, "C": 1.0},
    "logistic": {"C": 1.0},
    "hgb": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    "lightgbm": {
        "learning_rate": 0.05,
        "n_estimators": 300,
        "num_leaves": 31,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
    },
}


def build_model(name: str, params: dict[str, Any] | None = None, seed: int = 42) -> Pipeline:
    """Preprocessing + estimator as one pipeline (the unit that gets versioned and served)."""
    if name not in SEARCH_SPACES:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(SEARCH_SPACES)}")
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "scorecard":
        estimator = ScorecardModel(**p)
    elif name == "logistic":
        estimator = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=p["C"], max_iter=2000),
        )
    elif name == "hgb":
        estimator = HistGradientBoostingClassifier(**p, random_state=seed)
    else:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'credit-risk-mlops[lightgbm]'") from exc
        estimator = LGBMClassifier(
            **p, subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1
        )
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])
