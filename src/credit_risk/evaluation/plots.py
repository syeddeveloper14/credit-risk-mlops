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
