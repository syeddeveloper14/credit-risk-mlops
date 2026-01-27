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


def test_decile_table():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = decile_table(y, p)
    assert len(t) == 10 and t["cum_bad_capture"].iloc[-1] == pytest.approx(1.0)
    assert t["bad_rate"].iloc[0] > t["bad_rate"].iloc[-1]


def test_fairness_slices():
    age = pd.Series([25] * 400 + [50] * 400 + [70] * 400)
    rng = np.random.default_rng(2)
    p = np.r_[rng.uniform(0.1, 0.4, 400), rng.uniform(0, 0.1, 400), rng.uniform(0, 0.1, 400)]
    y = (rng.random(1200) < p).astype(int)
    table = slice_metrics(age, y, p, decline_above_pd=0.2)
    assert list(table.index) == ["<30", "45-61", "62+"]
    summary = fairness_summary(table)
    assert summary["approval_ratio"] < 0.8  # young band is approved far less often


def test_tuning_never_worse_than_defaults(splits):
    X, y = splits["train"][FEATURES], splits["train"][TARGET]
    default_score = cv_auc("logistic", DEFAULT_PARAMS["logistic"], X, y, folds=2, seed=0)
    for backend in ("random", "optuna"):
        if backend == "optuna":
            pytest.importorskip("optuna")
        result = tune("logistic", X, y, n_trials=4, cv_folds=2, seed=0, backend=backend)
        assert result.backend == backend
        assert result.best_score >= default_score - 1e-12
        assert len(result.history) >= 1


def test_tuning_with_one_trial_returns_defaults(splits):
    X, y = splits["train"][FEATURES], splits["train"][TARGET]
    result = tune("scorecard", X, y, n_trials=1, cv_folds=2)
    assert result.backend == "defaults" and result.best_params == DEFAULT_PARAMS["scorecard"]
