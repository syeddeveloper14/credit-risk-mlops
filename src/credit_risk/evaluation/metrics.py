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
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1) / len(
        actual
    )
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
        "ece": expected_calibration_error(y, p),
        "default_rate": float(y.mean()),
        "mean_pd": float(p.mean()),
    }


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Risk-ranked deciles (1 = riskiest) with bad rate, lift and cumulative capture."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    frame["decile"] = np.arange(len(frame)) * n_bins // len(frame) + 1
    t = frame.groupby("decile").agg(n=("y", "size"), bads=("y", "sum"), mean_pd=("p", "mean"))
    t["bad_rate"] = t["bads"] / t["n"]
    t["lift"] = t["bad_rate"] / frame["y"].mean()
    t["cum_bad_capture"] = t["bads"].cumsum() / frame["y"].sum()
    return t
