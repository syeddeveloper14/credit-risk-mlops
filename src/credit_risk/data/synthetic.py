"""Synthetic applicants in the original Kaggle schema.

Reproduces the quirks that make the real dataset a good lifecycle exercise: ~20% missing
income, debt ratios that become dollar amounts when income is missing, utilisation outliers
far above 1, the 96/98 sentinel codes in past-due counts, and a ~6-7% default rate.

``drift`` shifts the applicant population (lower income, higher utilisation, more
delinquency), which is used to demonstrate monitoring and retraining triggers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_applicants(n: int = 30_000, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(drift * 0.6, 1.0, n)  # latent credit risk (higher = riskier)

    age = np.clip(np.round(52 - 5 * z + rng.normal(0, 14, n)), 21, 99).astype(int)
    income = np.exp(rng.normal(np.log(5400) - 0.25 * z - 0.15 * drift, 0.7, n)).round()
