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
