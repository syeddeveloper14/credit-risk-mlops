"""Weight-of-evidence binning with monotonicity constraints and information value.

WoE(bin) = ln( share of goods in bin / share of bads in bin ), so higher WoE = lower risk.
Missing values get their own bin. Bins are merged until each holds at least
``min_bin_share`` of the data and the bad rate moves in one direction across bins, which is
what makes a scorecard explainable ("more utilisation never lowers risk").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class ColumnBins:
    edges: np.ndarray  # interior cut points; bin i covers (edges[i-1], edges[i]]
    woe: np.ndarray
    woe_missing: float
    iv: float
    table: pd.DataFrame

    def assign(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, x, side="left")

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = self.woe[self.assign(np.nan_to_num(x, nan=0.0))]
        return np.where(np.isnan(x), self.woe_missing, out)


def _counts(bins: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.bincount(bins, minlength=n_bins).astype(float)
    bads = np.bincount(bins, weights=y, minlength=n_bins).astype(float)
    return total, bads


def _is_monotonic(rates: np.ndarray, increasing: bool) -> bool:
    d = np.diff(rates)
    return bool(np.all(d >= 0) if increasing else np.all(d <= 0))


class WoEBinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_bins: int = 8,
        min_bin_share: float = 0.05,
        monotonic: bool = True,
        min_iv: float = 0.0,
        smoothing: float = 0.5,
    ):
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.monotonic = monotonic
