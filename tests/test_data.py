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

