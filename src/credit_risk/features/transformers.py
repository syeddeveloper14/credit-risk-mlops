"""Scikit-learn transformers for the credit features.

All transformers take and return pandas DataFrames, so column names survive the pipeline
(needed for scorecards, reason codes and drift monitoring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..data.schema import PAST_DUE, SENTINEL_VALUES


class _FrameTransformer(BaseEstimator, TransformerMixin):
    def _check(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} expects a pandas DataFrame")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class SentinelToNaN(_FrameTransformer):
    """Replace the 96/98 codes in past-due counts with NaN and add a flag column."""

    def __init__(self, columns=tuple(PAST_DUE), sentinels=SENTINEL_VALUES, flag="past_due_code"):
        self.columns = columns
        self.sentinels = sentinels
        self.flag = flag

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check(X)
        self.feature_names_out_ = [*X.columns, self.flag]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check(X).copy()
        cols = list(self.columns)
        mask = X[cols].isin(self.sentinels)
        X[cols] = X[cols].mask(mask)
        X[self.flag] = mask.any(axis=1).astype(int)
        return X


class CreditFeatures(_FrameTransformer):
    """Domain features. Stateless: nothing is learned from the data."""

    def fit(self, X: pd.DataFrame, y=None):
