"""Lifecycle stages: ingest -> validate -> split -> train -> evaluate -> register -> promote."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .. import __version__
from ..config import Config
from ..data import DataContractError, ingest, stratified_split, validate
from ..data.schema import FEATURES, TARGET
from ..evaluation import (
    classification_metrics,
    decile_table,
    fairness_summary,
    plots,
    psi,
    slice_metrics,
)
from ..models.scorecard import ScorecardModel, ScoreScaling
from ..monitoring import build_reference
from ..registry import ModelRegistry, decide_promotion
from ..tracking import make_tracker
from ..training import train_and_select
from .runner import Stage

log = logging.getLogger(__name__)

DATASET = "dataset.csv"
SPLITS = ("train", "valid", "test")


@dataclass
class Context:
    cfg: Config
    root: Path = field(init=False)
    tracker: Any = field(init=False)
    registry: ModelRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.cfg.paths.root)
        self.tracker = make_tracker(self.cfg.tracking, self.root)
        self.registry = ModelRegistry(
            self.root / self.cfg.registry.root, self.cfg.registry.model_name
        )

    def path(self, section: str, *parts: str) -> Path:
        return self.cfg.paths.resolve(section).joinpath(*parts)

    def rel(self, section: str, *parts: str) -> str:
        return str(Path(getattr(self.cfg.paths, section)).joinpath(*parts))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def _read_split(ctx: Context, name: str) -> pd.DataFrame:
    return pd.read_csv(ctx.path("processed", f"{name}.csv"))


# ---------------------------------------------------------------------- stages
def stage_ingest(ctx: Context) -> dict:
    df = ingest(ctx.cfg.data)
    out = ctx.path("interim", DATASET)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return {"rows": len(df), "default_rate": float(df[TARGET].mean())}


def stage_validate(ctx: Context) -> dict:
    df = pd.read_csv(ctx.path("interim", DATASET))
    report = validate(df, ctx.cfg.validation)
    report.save(ctx.path("reports", "validation.json"))
    if not report.passed:
        raise DataContractError(report)
    return {"passed": True, "warnings": [c.name for c in report.failures]}


def stage_split(ctx: Context) -> dict:
    df = pd.read_csv(ctx.path("interim", DATASET))
    splits = stratified_split(df, ctx.cfg.split)
    ctx.path("processed").mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(ctx.path("processed", f"{name}.csv"), index=False)
    return {name: len(part) for name, part in splits.items()}


def stage_train(ctx: Context) -> dict:
    train, valid = _read_split(ctx, "train"), _read_split(ctx, "valid")
    outcome = train_and_select(ctx.cfg, train, valid, ctx.tracker)
    ctx.path("artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(outcome.model, ctx.path("artifacts", "model.joblib"))
    board = outcome.leaderboard()
    summary = {
        "selected_model": outcome.best.name,
        "params": outcome.best.params,
        "valid_metrics": outcome.best.valid_metrics,
        "leaderboard": board.to_dict(orient="records"),
        "package_version": __version__,
    }
    _write_json(ctx.path("reports", "training.json"), summary)
    return {"selected_model": outcome.best.name, "valid_auc": outcome.best.valid_metrics["roc_auc"]}


def evaluate_model(
    model, train: pd.DataFrame, test: pd.DataFrame, cfg: Config
) -> tuple[dict, dict[str, pd.DataFrame]]:
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    p_train = model.predict_proba(train[FEATURES])[:, 1]
    metrics = classification_metrics(test[TARGET], p_test)
    metrics["score_psi"] = psi(p_train, p_test)
    slices = slice_metrics(
        test["age"], test[TARGET].to_numpy(), p_test, cfg.scoring.decline_above_pd
    )
    metrics.update(fairness_summary(slices))
    tables = {"deciles": decile_table(test[TARGET], p_test), "age_slices": slices}
    return metrics, tables


def _md_table(df: pd.DataFrame) -> str:
    frame = df.reset_index()
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append(
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


def stage_evaluate(ctx: Context) -> dict:
    cfg = ctx.cfg
    model = joblib.load(ctx.path("artifacts", "model.joblib"))
    train, valid, test = (_read_split(ctx, s) for s in SPLITS)
    train_full = pd.concat([train, valid])
    metrics, tables = evaluate_model(model, train_full, test, cfg)
    _write_json(ctx.path("reports", "evaluation.json"), metrics)

    figs = ctx.path("reports", "figures")
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    scaling = ScoreScaling(cfg.scoring.pdo, cfg.scoring.base_score, cfg.scoring.base_odds)
    plots.roc_plot({"selected model": (test[TARGET].to_numpy(), p_test)}, figs / "roc.png")
    plots.calibration_plot(test[TARGET].to_numpy(), p_test, figs / "calibration.png")
    plots.score_distribution_plot(
        test[TARGET].to_numpy(), scaling.score_from_pd(p_test), figs / "scores.png"
    )

    training = json.loads(ctx.path("reports", "training.json").read_text())
    board = pd.DataFrame(training["leaderboard"]).set_index("model")
    extra = ""
    estimator = model.named_steps["model"]
    if isinstance(estimator, ScorecardModel):
        pts = estimator.points_table(scaling)
        pts.to_csv(ctx.path("reports", "scorecard_points.csv"), index=False)
        extra = (
