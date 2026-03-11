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
    drift: pd.DataFrame,
    performance: dict[str, Any] | None = None,
    max_alerting_features: int = 1,
) -> RetrainDecision:
    reasons = []
    alerts = drift[(drift["status"] == "alert") & (drift["feature"] != "__score__")]
    if len(alerts) >= max_alerting_features:
        reasons.append(f"{len(alerts)} feature(s) with PSI alert: {', '.join(alerts['feature'])}")
    score = drift[drift["feature"] == "__score__"]
    if len(score) and score["status"].iloc[0] == "alert":
        reasons.append(f"score distribution shifted (PSI {score['psi'].iloc[0]:.3f})")
    if performance and performance.get("evaluated") and performance.get("degraded"):
        reasons.append(
            f"AUC dropped by {performance['auc_drop']:.3f} "
            f"({performance['reference_auc']:.3f} -> {performance['auc']:.3f})"
        )
    return RetrainDecision(retrain=bool(reasons), reasons=reasons)
