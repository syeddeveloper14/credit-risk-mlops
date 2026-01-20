"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE, GREY = "#2563eb", "#f97316", "#6b7280"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def roc_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(frameon=False)
    return _save(fig, path)


def calibration_plot(y: np.ndarray, p: np.ndarray, path: Path, n_bins: int = 10) -> Path:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    g = frame.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = max(g["pred"].max(), g["obs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, linestyle="--", linewidth=1)
    ax.plot(g["pred"], g["obs"], marker="o", color=BLUE)
    ax.set_xlabel("predicted PD")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration (equal-count bins)")
    return _save(fig, path)


def score_distribution_plot(y: np.ndarray, scores: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 50)
    ax.hist(scores[y == 0], bins=bins, alpha=0.6, density=True, color=BLUE, label="good")
    ax.hist(scores[y == 1], bins=bins, alpha=0.6, density=True, color=ORANGE, label="default")
    ax.set_xlabel("credit score")
    ax.set_ylabel("density")
    ax.set_title("Score distribution by outcome")
    ax.legend(frameon=False)
    return _save(fig, path)
