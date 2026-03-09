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
        "n": int(len(X)),
        "features": {c: _profile_column(X[c].to_numpy(dtype=float), n_bins) for c in X.columns},
        "score": _profile_column(np.asarray(scores, dtype=float), n_bins),
    }


def drift_report(
    reference: dict[str, Any],
    X: pd.DataFrame,
    scores: np.ndarray | None = None,
    warn: float = 0.1,
    alert: float = 0.25,
) -> pd.DataFrame:
    rows = []
    items = list(reference["features"].items())
    if scores is not None:
        items.append(("__score__", reference["score"]))
    for name, ref in items:
        values = np.asarray(scores, float) if name == "__score__" else X[name].to_numpy(dtype=float)
        value = _psi(ref["proportions"], _proportions(values, ref["edges"]))
        rows.append(
            {
                "feature": name,
                "psi": value,
                "status": "alert" if value >= alert else "warn" if value >= warn else "ok",
                "missing_rate": float(np.isnan(values).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def performance_check(
    y: np.ndarray, p: np.ndarray, reference_auc: float, max_drop: float, min_labels: int
) -> dict[str, Any]:
    y = np.asarray(y)
    if len(y) < min_labels or len(np.unique(y)) < 2:
        return {"evaluated": False, "reason": f"need >= {min_labels} labels with both classes"}
    auc = float(roc_auc_score(y, p))
    return {
        "evaluated": True,
        "auc": auc,
        "reference_auc": reference_auc,
        "auc_drop": reference_auc - auc,
        "degraded": reference_auc - auc > max_drop,
    }
