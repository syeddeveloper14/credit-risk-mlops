"""Credit-risk evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Max distance between the score distributions of bads and goods."""
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE over equal-count bins (equal-width bins are mostly empty at low default rates)."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    grouped = frame.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float(np.sum(grouped["n"] / len(frame) * np.abs(grouped["y"] - grouped["p"])))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of ``actual`` against quantile bins of ``expected``."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1) / len(
        expected
    )
