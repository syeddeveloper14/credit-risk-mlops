import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURES, TARGET
from credit_risk.features import (
    CreditFeatures,
    QuantileClipper,
    SentinelToNaN,
    WoEBinner,
    make_preprocessor,
)
from credit_risk.models import ScorecardModel, ScoreScaling, build_model


def row(**kw):
    base = {
        "revolving_utilization": 0.5,
        "age": 40,
        "late_30_59": 1,
        "debt_ratio": 0.3,
        "monthly_income": 5000.0,
        "open_credit_lines": 8,
        "late_90": 0,
        "real_estate_loans": 1,
        "late_60_89": 0,
        "dependents": 1.0,
    }
    base.update(kw)
    return pd.DataFrame([base])


def test_sentinels_become_missing_with_flag():
    X = pd.concat([row(late_30_59=98, late_60_89=98, late_90=98), row()], ignore_index=True)
    out = SentinelToNaN().fit_transform(X)
    assert out.loc[0, ["late_30_59", "late_60_89", "late_90"]].isna().all()
    assert out["past_due_code"].tolist() == [1, 0]
    assert out.loc[1, "late_30_59"] == 1


def test_credit_features():
    X = pd.concat([row(monthly_income=np.nan, debt_ratio=2500.0), row()], ignore_index=True)
    out = CreditFeatures().fit_transform(X)
    assert out.loc[0, "income_missing"] == 1 and np.isnan(out.loc[0, "debt_ratio_clean"])
    assert out.loc[1, "monthly_debt"] == pytest.approx(1500.0)
    assert out.loc[1, "income_per_person"] == pytest.approx(2500.0)
    assert out.loc[1, "total_past_due"] == 1
    assert out.loc[1, "lines_per_decade"] == pytest.approx(2.0)


def test_clipper_learns_bounds_on_fit_data_only():
    train = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    clipper = QuantileClipper(upper=0.99).fit(train)
    out = clipper.transform(pd.DataFrame({"x": [5000.0, 10.0]}))
    assert out["x"].iloc[0] == pytest.approx(train["x"].quantile(0.99))
    assert out["x"].iloc[1] == 10.0


def test_preprocessor_keeps_names(splits):
    prep = make_preprocessor().fit(splits["train"][FEATURES])
    out = prep.transform(splits["valid"][FEATURES])
    assert isinstance(out, pd.DataFrame) and "total_past_due" in out.columns
    assert len(out) == len(splits["valid"])

