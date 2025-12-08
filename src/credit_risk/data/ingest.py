"""Ingest raw data into the canonical schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataConfig
from .schema import ID, RAW_TO_CANONICAL
from .synthetic import simulate_applicants


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:  # the Kaggle CSV stores the row id in an unnamed first column
        df = df.set_index(unnamed[0])
    df = df.rename(columns=RAW_TO_CANONICAL)
    missing = [c for c in RAW_TO_CANONICAL.values() if c not in df.columns]
    # the target may be absent for scoring-only files
    missing = [c for c in missing if c != "serious_dlq_2yrs"]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df.index.name = ID
    df = df.reset_index()
    df[ID] = df[ID].astype(int)
    cols = [ID, *[c for c in RAW_TO_CANONICAL.values() if c in df.columns]]
    return df[cols]


def ingest(cfg: DataConfig) -> pd.DataFrame:
    if cfg.source == "synthetic":
        return to_canonical(simulate_applicants(cfg.n_synthetic, cfg.seed))
    if cfg.source == "file":
        path = Path(cfg.path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; see data/README.md or use data.source=synthetic"
            )
        return to_canonical(pd.read_csv(path))
    raise ValueError(f"Unknown data.source {cfg.source!r}")
