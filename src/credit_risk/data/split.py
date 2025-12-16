"""Stratified train / validation / test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import SplitConfig
from .schema import TARGET


