import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURES, TARGET
from credit_risk.evaluation import (
    classification_metrics,
    decile_table,
    expected_calibration_error,
    fairness_summary,
    ks_statistic,
    psi,
    slice_metrics,
)
from credit_risk.models import DEFAULT_PARAMS
from credit_risk.training import cv_auc, tune


def test_ks_and_gini():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, np.r_[np.zeros(50), np.ones(50)]) == 1.0
    m = classification_metrics(y, np.linspace(0.01, 0.99, 100))
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)


def test_ece_is_small_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.05


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=10_000)
    assert psi(a, rng.normal(size=10_000)) < 0.02
    assert psi(a, rng.normal(1.0, 1.0, size=10_000)) > 0.25
