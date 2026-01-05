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
