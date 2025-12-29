import numpy as np
import pandas as pd
import pytest

from credit_risk.config import SplitConfig
from credit_risk.data import (
    FEATURES,
    ID,
    TARGET,
    simulate_applicants,
    stratified_split,
    to_canonical,
    validate,
)


def test_to_canonical_handles_kaggle_layout():
    raw = simulate_applicants(50, seed=1).reset_index().rename(columns={"index": "Unnamed: 0"})
    df = to_canonical(raw)
    assert list(df.columns) == [ID, TARGET, *FEATURES]
    assert df[ID].tolist() == list(range(1, 51))


def test_to_canonical_without_target():
    raw = simulate_applicants(20, seed=1).drop(columns=["SeriousDlqin2yrs"])
    df = to_canonical(raw)
    assert TARGET not in df.columns and set(FEATURES) <= set(df.columns)


def test_synthetic_has_real_data_quirks(applicants):
    assert 0.03 < applicants[TARGET].mean() < 0.12
    assert 0.1 < applicants["monthly_income"].isna().mean() < 0.3
    assert applicants[["late_30_59", "late_90"]].isin([96, 98]).any().any()
    assert (applicants["revolving_utilization"] > 1.5).any()


def test_validation_passes(applicants):
    report = validate(applicants)
    assert report.passed
    # debt_ratio holds dollar amounts when income is missing: a warning, not a failure
    assert any("debt_ratio" in c.name for c in report.failures)


@pytest.mark.parametrize(
    "corrupt, check",
    [
        (lambda d: d.assign(age=-d["age"]), "non-negative"),
        (lambda d: pd.concat([d, d.head(3)]), "unique applicant ids"),
        (lambda d: d.assign(serious_dlq_2yrs=2), "target is binary"),
        (lambda d: d.drop(columns=["late_90"]), "required columns present"),
        (lambda d: d.assign(dependents=np.nan), "dependents: null rate"),
    ],
)
def test_validation_blocks_bad_data(applicants, corrupt, check):
    report = validate(corrupt(applicants.copy()))
    assert not report.passed
    assert any(check in c.name for c in report.failures if c.blocking)


def test_stratified_split(applicants):
    parts = stratified_split(applicants, SplitConfig(valid_size=0.2, test_size=0.2, seed=1))
    sizes = {k: len(v) for k, v in parts.items()}
    assert sum(sizes.values()) == len(applicants)
    assert abs(sizes["test"] - 0.2 * len(applicants)) <= 1
    rates = [v[TARGET].mean() for v in parts.values()]
    assert max(rates) - min(rates) < 0.005
    ids = [set(v[ID]) for v in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    with pytest.raises(ValueError):
        stratified_split(applicants, SplitConfig(valid_size=0.6, test_size=0.5))
