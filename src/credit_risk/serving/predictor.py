"""Turn a fitted pipeline into credit decisions: PD, score, decision and reason codes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import ScoringConfig
from ..data.schema import FEATURES, ID
from ..models.scorecard import ScorecardModel, ScoreScaling
from ..registry import ModelRegistry

REASON_TEXT: dict[str, str] = {
    "revolving_utilization": "Revolving balances are high relative to credit limits",
    "age": "Age profile of the applicant",
    "late_30_59": "Recent 30-59 day delinquencies",
    "late_60_89": "Recent 60-89 day delinquencies",
    "late_90": "Serious delinquency (90+ days late)",
    "total_past_due": "Number of past-due events",
    "any_90_late": "Serious delinquency on record",
    "debt_ratio": "Debt payments are high relative to income",
    "debt_ratio_clean": "Debt payments are high relative to income",
    "monthly_debt": "Total monthly debt payments",
    "monthly_income": "Income level",
    "income_per_person": "Income relative to household size",
    "income_missing": "Income not provided",
    "open_credit_lines": "Number of open credit lines",
    "lines_per_decade": "Number of credit lines relative to age",
    "real_estate_loans": "Number of real estate loans",
    "dependents": "Number of dependents",
    "past_due_code": "Unusual delinquency record",
    "utilization_over_1": "Revolving balances exceed credit limits",
}


class CreditRiskPredictor:
    def __init__(
        self,
        model: Pipeline,
        scoring: ScoringConfig | None = None,
        version: str = "dev",
        reference_goods: dict[str, float] | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.scoring = scoring or ScoringConfig()
        self.scaling = ScoreScaling(
            self.scoring.pdo, self.scoring.base_score, self.scoring.base_odds
        )
        self.version = version
        self.reference_goods = reference_goods or {}
        self.log_path = Path(log_path) if log_path else None

    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
        scoring: ScoringConfig,
        stage: str = "Production",
        log_path: str | Path | None = None,
    ) -> CreditRiskPredictor:
        mv = registry.latest(stage)
        if mv is None:
            raise LookupError(f"No model in stage {stage!r}; run the pipeline first")
        model = registry.load_model(mv.version)
        goods = mv.reference.get("goods_median", {})
        return cls(model, scoring, f"{mv.name}:v{mv.version}", goods, log_path)

    @property
    def estimator(self):
        return self.model.named_steps["model"]

    def decisions(self, pd_bad: np.ndarray) -> np.ndarray:
        s = self.scoring
        return np.where(
            pd_bad >= s.decline_above_pd,
            "decline",
            np.where(pd_bad < s.approve_below_pd, "approve", "refer"),
        )

    def _reasons(self, X: pd.DataFrame, pd_bad: np.ndarray, top_k: int = 4) -> list[list[str]]:
        if isinstance(self.estimator, ScorecardModel):
            prepped = self.model.named_steps["prep"].transform(X)
            return self.estimator.reason_codes(prepped, self.scaling, top_k)
        # Model-agnostic fallback: replace each raw input with the median of good applicants
        # and keep the inputs whose replacement lowers PD the most.
        features = [f for f in FEATURES if f in self.reference_goods]
        impacts = np.zeros((len(X), len(features)))
        for j, f in enumerate(features):
            X_alt = X.copy()
            X_alt[f] = self.reference_goods[f]
            impacts[:, j] = pd_bad - self.model.predict_proba(X_alt)[:, 1]
        out = []
        for row in impacts:
            order = np.argsort(-row)[:top_k]
            out.append([features[i] for i in order if row[i] > 0.001])
        return out

    def predict(self, df: pd.DataFrame, with_reasons: bool = True) -> pd.DataFrame:
        X = df[FEATURES].astype(float)
        pd_bad = self.model.predict_proba(X)[:, 1]
        decision = self.decisions(pd_bad)
        out = pd.DataFrame(
            {
                "probability_of_default": pd_bad,
                "credit_score": np.round(self.scaling.score_from_pd(pd_bad)).astype(int),
                "decision": decision,
            },
            index=df.index,
        )
        if ID in df.columns:
            out.insert(0, ID, df[ID].to_numpy())
        if with_reasons:
            reasons = self._reasons(X, pd_bad)
            # Reasons are only meaningful (and legally required) for non-approved applicants.
            out["reasons"] = [
                r if d != "approve" else [] for r, d in zip(reasons, decision, strict=True)
            ]
        out["model_version"] = self.version
        if self.log_path is not None:
            self._log(df, out)
        return out

    def _log(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        """Append predictions to a JSONL log: the raw material for monitoring."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(self.log_path, "a", encoding="utf-8") as fh:
            for (_, x), (_, y) in zip(inputs[FEATURES].iterrows(), outputs.iterrows(), strict=True):
                record: dict[str, Any] = {
                    "time": now,
                    "model_version": self.version,
                    "inputs": {k: (None if pd.isna(v) else float(v)) for k, v in x.items()},
                    "probability_of_default": float(y["probability_of_default"]),
                    "decision": y["decision"],
                }
                fh.write(json.dumps(record) + "\n")
