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
    income_missing = rng.random(n) < 0.19
    dependents = rng.poisson(0.75 + 0.1 * (age < 50), n).astype(float)
    dependents_missing = income_missing & (rng.random(n) < 0.13)

    utilization = _sigmoid(rng.normal(-1.0 + 1.1 * z + 0.4 * drift, 1.2, n))
    outliers = rng.random(n) < 0.002
    utilization[outliers] = rng.uniform(1.5, 5000, outliers.sum())

    debt_ratio = np.exp(rng.normal(np.log(0.3) + 0.15 * z, 0.6, n))
    # Known quirk of the public data: without an income the "ratio" is a dollar amount.
    debt_ratio[income_missing] = rng.gamma(1.2, 1200, income_missing.sum())

    lam = np.exp(-2.0 + 0.9 * z + 0.3 * drift)
    late_30_59 = rng.poisson(lam * 1.0)
    late_60_89 = rng.poisson(lam * 0.35)
    late_90 = rng.poisson(lam * 0.4)
    sentinel = rng.random(n) < 0.0018
    code = rng.choice([96, 98], size=n)
    for col in (late_30_59, late_60_89, late_90):
        col[sentinel] = code[sentinel]

    open_lines = rng.poisson(np.clip(8.5 - 0.6 * z, 0.5, None))
    real_estate = rng.poisson(np.clip(1.0 - 0.2 * z, 0.05, None))

    logit = (
        -3.4
