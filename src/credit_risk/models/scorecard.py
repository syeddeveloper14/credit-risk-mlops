"""Points-based credit scorecard: WoE binning + logistic regression + score scaling.

Score scaling (industry convention):

    score = offset + factor * ln(odds_good)
    factor = PDO / ln(2)                       # PDO = points to double the odds
    offset = base_score - factor * ln(base_odds)

With logit(P(bad)) = b0 + sum_i b_i * WoE_i, each characteristic contributes

    points_i = -factor * b_i * WoE_i + (offset - factor * b0) / n

so an applicant's score is the sum of their attribute points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from ..features.woe import WoEBinner


@dataclass(frozen=True)
class ScoreScaling:
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_pd(self, pd_bad: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(pd_bad, dtype=float), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - p) / p)

    def pd_from_score(self, score: np.ndarray) -> np.ndarray:
        odds_good = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1 / (1 + odds_good)


class ScorecardModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_bins: int = 8,
        min_bin_share: float = 0.05,
        min_iv: float = 0.02,
        C: float = 1.0,
    ):
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.min_iv = min_iv
        self.C = C

    def fit(self, X: pd.DataFrame, y) -> ScorecardModel:
        self.binner_ = WoEBinner(
            max_bins=self.max_bins, min_bin_share=self.min_bin_share, min_iv=self.min_iv
        ).fit(X, y)
        W = self.binner_.transform(X)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(W, np.asarray(y))
        self.classes_ = self.lr_.classes_
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.binner_.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    # ----------------------------------------------------------------- points
    def _coef(self) -> dict[str, float]:
        return dict(zip(self.binner_.selected_, self.lr_.coef_[0], strict=True))

    def points(self, X: pd.DataFrame, scaling: ScoreScaling) -> pd.DataFrame:
        """Points per characteristic for each applicant; row sums equal the score."""
        W = self.binner_.transform(X)
        coef = self._coef()
        n = len(coef)
        base = (scaling.offset - scaling.factor * self.lr_.intercept_[0]) / n
        return pd.DataFrame(
            {c: -scaling.factor * coef[c] * W[c] + base for c in coef}, index=X.index
        )

    def credit_score(self, X: pd.DataFrame, scaling: ScoreScaling) -> np.ndarray:
        return self.points(X, scaling).sum(axis=1).to_numpy()

    def points_table(self, scaling: ScoreScaling) -> pd.DataFrame:
        coef = self._coef()
        n = len(coef)
        base = (scaling.offset - scaling.factor * self.lr_.intercept_[0]) / n
        frames = []
        for col, b in coef.items():
            t = self.binner_.bins_[col].table.copy()
            t.insert(0, "characteristic", col)
            t["points"] = (-scaling.factor * b * t["woe"] + base).round(1)
            frames.append(t)
        return pd.concat(frames, ignore_index=True)

    def reason_codes(
        self, X: pd.DataFrame, scaling: ScoreScaling, top_k: int = 4
    ) -> list[list[str]]:
        """Adverse-action style reasons: characteristics furthest below their best points."""
        pts = self.points(X, scaling)
        table = self.points_table(scaling)
        best = table.groupby("characteristic")["points"].max()
        shortfall = best[pts.columns].to_numpy()[None, :] - pts.to_numpy()
        out = []
        for row in shortfall:
            order = np.argsort(-row)[:top_k]
            out.append([pts.columns[i] for i in order if row[i] > 0.5])
        return out
