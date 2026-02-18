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
    results = run_pipeline(cfg, until="split")
    assert [r.name for r in results] == ["ingest", "validate", "split"]
    again = run_pipeline(cfg, until="split", force=["split"])
    assert [r.status for r in again] == ["cached", "cached", "ran"]


def test_topological_order_and_cycles():
    a = Stage("a", lambda c: {})
    b = Stage("b", lambda c: {}, deps=["a"])
    c = Stage("c", lambda c: {}, deps=["b"])
    assert [s.name for s in topological_order([c, b, a])] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        topological_order(
            [Stage("x", lambda c: {}, deps=["y"]), Stage("y", lambda c: {}, deps=["x"])]
        )


def test_promotion_gates():
    gates = PromotionConfig()
    assert decide_promotion(GOOD, None, gates).promote
    assert decide_promotion(GOOD, {"roc_auc": 0.79}, gates).promote
    assert not decide_promotion(GOOD, {"roc_auc": 0.7995}, gates).promote
    for key, bad in [
        ("ece", 0.2),
        ("score_psi", 0.5),
        ("age_auc_gap", 0.3),
        ("approval_ratio", 0.5),
        ("roc_auc", 0.6),
    ]:
        decision = decide_promotion({**GOOD, key: bad}, None, gates)
        assert not decision.promote, key
        assert "DO NOT PROMOTE" in decision.summary()


def test_registry_stages(tmp_path):
    reg = ModelRegistry(tmp_path, "m")
    v1 = reg.register({"w": 1}, "logistic", {"roc_auc": 0.7}, {}, ["x"])
    v2 = reg.register({"w": 2}, "hgb", {"roc_auc": 0.8}, {}, ["x"])
    assert (v1.version, v2.version) == (1, 2)
    reg.transition(1, "Production")
    reg.transition(2, "Production")
    assert reg.get(1).stage == "Archived" and reg.latest("Production").version == 2
    assert reg.load_model(2) == {"w": 2}
    assert len(reg.get(1).history) == 3
    assert list(reg.list_versions()["stage"]) == ["Archived", "Production"]
    with pytest.raises(ValueError):
        reg.transition(1, "Live")
    with pytest.raises(KeyError):
        reg.get(9)


def test_local_tracker(tmp_path):
    tracker = LocalTracker(tmp_path, "exp")
    with tracker.start_run("ok", tags={"k": "v"}):
        tracker.log_params({"model": "x", "nested": {"a": 1}})
        tracker.log_metrics({"auc": 0.7}, step=0)
        tracker.log_metrics({"auc": 0.8}, step=1)
    with pytest.raises(RuntimeError), tracker.start_run("bad"):
        raise RuntimeError("boom")
    runs = tracker.list_runs()
    assert list(runs["status"]) == ["FINISHED", "FAILED"]
    assert runs.iloc[0]["metric.auc"] == 0.8 and runs.iloc[0]["param.nested.a"] == 1
