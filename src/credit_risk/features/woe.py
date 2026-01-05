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
        self.min_iv = min_iv
        self.smoothing = smoothing

    # ------------------------------------------------------------------ fitting
    def _fit_column(self, x: np.ndarray, y: np.ndarray) -> ColumnBins:
        nan = np.isnan(x)
        xv, yv = x[~nan], y[~nan]
        if len(xv) == 0:
            edges = np.array([])
        else:
            quantiles = np.linspace(0, 1, self.max_bins + 1)[1:-1]
            edges = np.unique(np.quantile(xv, quantiles))
            # Drop edges equal to the maximum: they would create an empty last bin.
            edges = edges[edges < xv.max()]

        min_count = self.min_bin_share * len(x)
        while len(edges) > 0:
            bins = np.searchsorted(edges, xv, side="left")
            total, bads = _counts(bins, yv, len(edges) + 1)
            small = np.flatnonzero(total < min_count)
            if len(small):
                i = int(small[np.argmin(total[small])])
                # merge with the neighbour (drop the edge on the side with fewer rows)
                if i == 0:
                    edges = np.delete(edges, 0)
                elif i == len(total) - 1:
                    edges = np.delete(edges, i - 1)
                else:
                    edges = np.delete(edges, i - 1 if total[i - 1] <= total[i + 1] else i)
                continue
            if not self.monotonic:
                break
            rates = bads / np.maximum(total, 1)
            increasing = (
                np.corrcoef(np.arange(len(rates)), rates)[0, 1] >= 0 if len(rates) > 2 else True
            )
            if _is_monotonic(rates, increasing):
                break
            d = np.diff(rates)
            violations = np.flatnonzero(d < 0) if increasing else np.flatnonzero(d > 0)
            edges = np.delete(edges, int(violations[0]))

        n_bins = len(edges) + 1
        bins = np.searchsorted(edges, xv, side="left")
        total, bads = _counts(bins, yv, n_bins)
        goods = total - bads
        miss_total, miss_bads = float(nan.sum()), float(y[nan].sum())
        miss_goods = miss_total - miss_bads

        s = self.smoothing
        k = n_bins + (1 if miss_total else 0)
        all_goods = goods.sum() + miss_goods
        all_bads = bads.sum() + miss_bads
        dist_g = (goods + s) / (all_goods + s * k)
        dist_b = (bads + s) / (all_bads + s * k)
        woe = np.log(dist_g / dist_b)
        iv = float(np.sum((dist_g - dist_b) * woe))
        woe_missing = 0.0
        if miss_total:
            mg = (miss_goods + s) / (all_goods + s * k)
            mb = (miss_bads + s) / (all_bads + s * k)
            woe_missing = float(np.log(mg / mb))
            iv += float((mg - mb) * woe_missing)

        lows = np.r_[-np.inf, edges]
        highs = np.r_[edges, np.inf]
        labels = [f"({lo:.4g}, {hi:.4g}]" for lo, hi in zip(lows, highs, strict=True)]
        table = pd.DataFrame(
            {
                "bin": labels,
                "count": total,
                "bads": bads,
                "bad_rate": bads / np.maximum(total, 1),
                "woe": woe,
            }
        )
        if miss_total:
            table.loc[len(table)] = [
                "missing",
                miss_total,
                miss_bads,
                miss_bads / miss_total,
                woe_missing,
            ]
        return ColumnBins(edges=edges, woe=woe, woe_missing=woe_missing, iv=iv, table=table)

    def fit(self, X: pd.DataFrame, y) -> WoEBinner:
        y = np.asarray(y, dtype=float)
        self.bins_ = {c: self._fit_column(X[c].to_numpy(dtype=float), y) for c in X.columns}
        self.iv_ = {c: b.iv for c, b in self.bins_.items()}
        self.selected_ = [c for c in X.columns if self.iv_[c] >= self.min_iv]
        if not self.selected_:
            raise ValueError("No feature reaches min_iv; lower the threshold")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {c: self.bins_[c].transform(X[c].to_numpy(dtype=float)) for c in self.selected_},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)

    def iv_table(self) -> pd.DataFrame:
        frame = pd.DataFrame({"iv": self.iv_}).sort_values("iv", ascending=False)
        frame["selected"] = frame.index.isin(self.selected_)
        frame["strength"] = pd.cut(
            frame["iv"],
            [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "suspicious"],
        )
        return frame
