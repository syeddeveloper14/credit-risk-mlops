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
    max_age_auc_gap: float = 0.10
    min_approval_ratio: float = 0.80


@dataclass
class TrackingConfig:
    backend: str = "local"  # "local" or "mlflow"
    uri: str = "mlruns_local"
    experiment: str = "credit-risk"


@dataclass
class RegistryConfig:
    root: str = "model_registry"
    model_name: str = "credit-default"


@dataclass
class MonitoringConfig:
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    max_auc_drop: float = 0.03
    min_labels_for_performance: int = 500
    prediction_log: str = "logs/predictions.jsonl"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def section(self, name: str) -> dict[str, Any]:
        return dataclasses.asdict(getattr(self, name))


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in values:
            continue
        ftype, value = hints[f.name], values[f.name]
        if dataclasses.is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, value)
        elif typing.get_origin(ftype) is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return _build(Config, raw)
