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
