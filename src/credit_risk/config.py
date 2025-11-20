"""Pipeline configuration (YAML -> typed dataclasses)."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"  # "synthetic" or "file"
    path: str = "data/raw/cs-training.csv"
    n_synthetic: int = 30_000
    seed: int = 7


@dataclass
class PathsConfig:
    root: str = "."
    interim: str = "data/interim"
    processed: str = "data/processed"
    artifacts: str = "artifacts"
    reports: str = "reports"
    cache: str = ".pipeline_cache"

    def resolve(self, name: str) -> Path:
        return Path(self.root) / getattr(self, name)


@dataclass
class ValidationConfig:
    min_rows: int = 1_000
    target_rate_range: tuple[float, float] = (0.01, 0.30)
    max_null_rate: dict[str, float] = field(
        default_factory=lambda: {"monthly_income": 0.30, "dependents": 0.05}
    )
    max_out_of_range_rate: float = 0.01


@dataclass
class SplitConfig:
    valid_size: float = 0.2
    test_size: float = 0.2
    seed: int = 42


@dataclass
class HPOConfig:
    n_trials: int = 25
    cv_folds: int = 3
    timeout_s: int | None = 600
    seed: int = 42


@dataclass
class ModelsConfig:
    candidates: list[str] = field(default_factory=lambda: ["scorecard", "logistic", "hgb"])
    selection_metric: str = "roc_auc"


@dataclass
class ScoringConfig:
    pdo: float = 20.0  # points to double the odds
    base_score: float = 600.0
    base_odds: float = 50.0  # good:bad odds at base_score
    approve_below_pd: float = 0.05
    decline_above_pd: float = 0.20


@dataclass
class PromotionConfig:
    min_auc: float = 0.75
    min_auc_improvement: float = 0.002
    max_ece: float = 0.03
    max_score_psi: float = 0.10
