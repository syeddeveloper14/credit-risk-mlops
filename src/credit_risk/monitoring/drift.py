"""Input drift (feature PSI), output drift (score PSI) and delayed-label performance checks.

The reference profile is computed once from the training data and stored in the registry
with the model version, so monitoring jobs need only the new batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-4


def _profile_column(values: np.ndarray, n_bins: int) -> dict[str, Any]:
    finite = values[~np.isnan(values)]
    edges = (
        np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])).tolist()
        if len(finite)
        else []
    )
    return {"edges": edges, "proportions": _proportions(values, edges).tolist()}


def _proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    nan = np.isnan(values)
    counts = np.bincount(
        np.searchsorted(edges, values[~nan], side="right"), minlength=len(edges) + 1
    )
    return np.r_[counts, nan.sum()] / max(len(values), 1)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(X: pd.DataFrame, scores: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    return {
