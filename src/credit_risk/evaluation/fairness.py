"""Slice-level checks by age band.

Age is both a model input and a characteristic protected in many credit regulations (for
example, ECOA / Regulation B in the US restricts how applicants aged 62+ may be treated).
These slices make disparities visible before a model is promoted; they do not replace a
legal review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BANDS = [0, 30, 45, 62, 200]
AGE_LABELS = ["<30", "30-44", "45-61", "62+"]


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=AGE_BANDS, labels=AGE_LABELS, right=False)


def slice_metrics(
    age: pd.Series, y: np.ndarray, p: np.ndarray, decline_above_pd: float
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"band": age_band(age).reset_index(drop=True), "y": np.asarray(y), "p": np.asarray(p)}
    )  # keep the categorical dtype so bands stay in age order
    rows = []
    for band, g in frame.groupby("band", observed=True):
        auc = roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan
        rows.append(
            {
                "age_band": str(band),
                "n": len(g),
                "default_rate": g["y"].mean(),
                "mean_pd": g["p"].mean(),
                "roc_auc": auc,
                "approval_rate": float((g["p"] < decline_above_pd).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("age_band")


def fairness_summary(table: pd.DataFrame, min_group_size: int = 200) -> dict[str, float]:
    """Gaps across bands with enough applicants to be meaningful."""
    t = table[table["n"] >= min_group_size]
    aucs = t["roc_auc"].dropna()
    approvals = t["approval_rate"]
    return {
        "age_auc_gap": float(aucs.max() - aucs.min()) if len(aucs) > 1 else 0.0,
        "approval_ratio": float(approvals.min() / approvals.max()) if approvals.max() > 0 else 0.0,
        "calibration_gap": float((t["mean_pd"] - t["default_rate"]).abs().max()) if len(t) else 0.0,
    }
