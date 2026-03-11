"""Retraining policy: turn monitoring signals into a yes/no decision with reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)


def retrain_decision(
