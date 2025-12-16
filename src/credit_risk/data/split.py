"""Stratified train / validation / test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import SplitConfig
from .schema import TARGET


def stratified_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or SplitConfig()
    if cfg.valid_size + cfg.test_size >= 1:
        raise ValueError("valid_size + test_size must be < 1")
    rest, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[TARGET], random_state=cfg.seed
    )
    valid_share = cfg.valid_size / (1 - cfg.test_size)
    train, valid = train_test_split(
        rest, test_size=valid_share, stratify=rest[TARGET], random_state=cfg.seed
    )
    return {"train": train, "valid": valid, "test": test}
