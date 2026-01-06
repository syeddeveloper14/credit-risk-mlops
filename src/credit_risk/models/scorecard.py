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
