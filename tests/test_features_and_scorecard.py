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


def test_woe_binning_is_monotonic_with_missing_bin(splits):
    X = make_preprocessor().fit_transform(splits["train"][FEATURES])
    y = splits["train"][TARGET]
    binner = WoEBinner(max_bins=8, min_bin_share=0.05).fit(X, y)
    for col in ("revolving_utilization", "age", "monthly_income"):
        rates = binner.bins_[col].table.query("bin != 'missing'")["bad_rate"].to_numpy()
        diffs = np.diff(rates)
        assert (diffs >= -1e-12).all() or (diffs <= 1e-12).all(), col
    assert "missing" in set(binner.bins_["monthly_income"].table["bin"])
    assert binner.iv_["revolving_utilization"] > binner.iv_["dependents"]
    W = binner.transform(X)
    assert not W.isna().any().any()


def test_woe_min_iv_selection():
    X = pd.DataFrame({"noise": np.random.default_rng(0).normal(size=500)})
    y = np.random.default_rng(1).integers(0, 2, 500)
    with pytest.raises(ValueError):
        WoEBinner(min_iv=0.5).fit(X, y)


def test_score_scaling():
    s = ScoreScaling(pdo=20, base_score=600, base_odds=50)
    assert s.score_from_pd(np.array([1 / 51]))[0] == pytest.approx(600)
    # doubling the good:bad odds adds exactly PDO points
    assert s.score_from_pd(np.array([1 / 101]))[0] == pytest.approx(620)
    p = np.array([0.01, 0.1, 0.5])
    assert s.pd_from_score(s.score_from_pd(p)) == pytest.approx(p)


def test_scorecard_points_sum_to_score(splits):
    pipe = build_model("scorecard").fit(splits["train"][FEATURES], splits["train"][TARGET])
    model: ScorecardModel = pipe.named_steps["model"]
    X = pipe.named_steps["prep"].transform(splits["valid"][FEATURES].head(50))
    scaling = ScoreScaling()
    from_points = model.credit_score(X, scaling)
    from_pd = scaling.score_from_pd(model.predict_proba(X)[:, 1])
    np.testing.assert_allclose(from_points, from_pd, atol=1e-6)
    table = model.points_table(scaling)
    assert set(table["characteristic"]) == set(model.binner_.selected_)


def test_reason_codes_name_weak_characteristics(splits):
    pipe = build_model("scorecard").fit(splits["train"][FEATURES], splits["train"][TARGET])
    model = pipe.named_steps["model"]
    risky = row(revolving_utilization=0.99, late_90=3, late_30_59=4, age=23)
    codes = model.reason_codes(pipe.named_steps["prep"].transform(risky), ScoreScaling())[0]
    assert "revolving_utilization" in codes
    assert len(codes) <= 4


@pytest.mark.parametrize("name", ["scorecard", "logistic", "hgb", "lightgbm"])
def test_all_candidates_fit(splits, name):
    if name == "lightgbm":
        pytest.importorskip("lightgbm")
    pipe = build_model(name).fit(splits["train"][FEATURES], splits["train"][TARGET])
    p = pipe.predict_proba(splits["valid"][FEATURES])[:, 1]
    assert p.shape == (len(splits["valid"]),) and ((p > 0) & (p < 1)).all()


def test_unknown_candidate():
    with pytest.raises(ValueError):
        build_model("svm")
