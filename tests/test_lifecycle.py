import json

import pytest

from credit_risk.config import PromotionConfig
from credit_risk.pipeline import Stage, run_pipeline, topological_order
from credit_risk.registry import ModelRegistry, decide_promotion
from credit_risk.tracking import LocalTracker
from tests.conftest import make_config

GOOD = {"roc_auc": 0.80, "ece": 0.01, "score_psi": 0.01, "age_auc_gap": 0.03, "approval_ratio": 0.9}


def test_first_run_promotes(pipeline_env):
    cfg, results = pipeline_env
    assert [r.status for r in results] == ["ran"] * 7
    registry = ModelRegistry(cfg.registry.root, cfg.registry.model_name)
    prod = registry.latest("Production")
    assert prod is not None and prod.version == 1
    assert prod.reference["features"] and prod.reference["goods_median"]
    root = cfg.paths.resolve("reports")
    for name in (
        "validation.json",
        "training.json",
        "evaluation.json",
        "evaluation.md",
        "promotion.json",
    ):
        assert (root / name).exists(), name


def test_second_run_is_cached(pipeline_env):
    cfg, _ = pipeline_env
    results = run_pipeline(cfg)
    status = {r.name: r.status for r in results}
    assert all(
        status[s] == "cached"
        for s in ("ingest", "validate", "split", "train", "evaluate", "register")
    )
    assert results[-1].summary["note"] == "already in Production"


def test_param_change_retrains_and_goes_through_gates(tmp_path):
    cfg = make_config(tmp_path, **{"models.candidates": ["logistic"], "hpo.n_trials": 1})
    run_pipeline(cfg)
    cfg2 = make_config(
        tmp_path, **{"models.candidates": ["logistic"], "hpo.n_trials": 1, "seed": 7}
    )
    results = {r.name: r for r in run_pipeline(cfg2)}
    assert results["ingest"].status == "cached"
    assert results["train"].status == "ran"
    assert results["register"].summary["version"] == 2
    promotion = json.loads((cfg.paths.resolve("reports") / "promotion.json").read_text())
    assert promotion["champion_version"] == 1
    # a near-identical retrain cannot beat the champion by the required margin
    assert promotion["promote"] is False
    registry = ModelRegistry(cfg.registry.root, cfg.registry.model_name)
    assert registry.get(2).stage == "Staging" and registry.latest("Production").version == 1


def test_until_and_force(tmp_path):
    cfg = make_config(tmp_path, **{"models.candidates": ["logistic"], "hpo.n_trials": 1})
