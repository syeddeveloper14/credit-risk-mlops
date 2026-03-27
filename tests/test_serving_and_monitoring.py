import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_risk.data import FEATURES, simulate_applicants, to_canonical
from credit_risk.monitoring import drift_report, performance_check, retrain_decision
from credit_risk.registry import ModelRegistry
from credit_risk.serving import CreditRiskPredictor
from credit_risk.serving.api import create_app

APPLICANT = {
    "applicant_id": 7,
    "revolving_utilization": 0.95,
    "age": 29,
    "late_30_59": 3,
    "debt_ratio": 0.9,
    "monthly_income": 2500,
    "open_credit_lines": 4,
    "late_90": 2,
    "real_estate_loans": 0,
    "late_60_89": 1,
    "dependents": 2,
}


@pytest.fixture(scope="module")
def production(pipeline_env):
    cfg, _ = pipeline_env
    registry = ModelRegistry(cfg.registry.root, cfg.registry.model_name)
    return cfg, registry, registry.latest("Production")


def test_predictor_outputs(production):
    cfg, registry, _ = production
    predictor = CreditRiskPredictor.from_registry(registry, cfg.scoring)
    batch = to_canonical(simulate_applicants(300, seed=21))
    out = predictor.predict(batch)
    assert set(out["decision"]) <= {"approve", "refer", "decline"}
    assert out["probability_of_default"].between(0, 1).all()
    # higher PD always means a lower score
    order = np.argsort(out["probability_of_default"].to_numpy())
    assert (np.diff(out["credit_score"].to_numpy()[order]) <= 0).all()
    approved = out[out["decision"] == "approve"]
    assert all(len(r) == 0 for r in approved["reasons"])
    flagged = out[out["decision"] != "approve"]
    assert all(len(r) > 0 for r in flagged["reasons"])


def test_model_agnostic_reasons(production, splits):
    cfg, registry, mv = production
    from credit_risk.models import build_model

    hgb = build_model("hgb").fit(splits["train"][FEATURES], splits["train"]["serious_dlq_2yrs"])
    predictor = CreditRiskPredictor(hgb, cfg.scoring, "hgb-test", mv.reference["goods_median"])
    import pandas as pd

    out = predictor.predict(pd.DataFrame([APPLICANT]))
    assert out["decision"].iloc[0] != "approve"
    assert len(out["reasons"].iloc[0]) > 0


def test_api(production, tmp_path):
    cfg, registry, mv = production
    client = TestClient(create_app(cfg=cfg))
    assert client.get("/health").json()["model_version"] == f"{mv.name}:v{mv.version}"
    info = client.get("/model").json()
    assert info["stage"] == "Production" and "roc_auc" in info["metrics"]

    r = client.post("/v1/predict", json=APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert body["applicant_id"] == 7 and body["decision"] in {"refer", "decline"}
    assert body["reasons"] and all(reason["text"] for reason in body["reasons"])

    assert client.post("/v1/predict", json={**APPLICANT, "age": 12}).status_code == 422
    missing_income = {**APPLICANT, "monthly_income": None, "dependents": None}
    assert client.post("/v1/predict", json=missing_income).status_code == 200

    batch = client.post(
        "/v1/predict/batch", json=[APPLICANT, {**APPLICANT, "applicant_id": 8}]
    ).json()
    assert [b["applicant_id"] for b in batch] == [7, 8]

    log_lines = Path(cfg.monitoring.prediction_log).read_text().strip().splitlines()
    assert len(log_lines) >= 4
    assert json.loads(log_lines[-1])["model_version"] == f"{mv.name}:v{mv.version}"


def test_drift_detection(production):
    cfg, registry, mv = production
    model = registry.load_model(mv.version)
    stable = to_canonical(simulate_applicants(5000, seed=11))
    shifted = to_canonical(simulate_applicants(5000, seed=12, drift=1.5))
    for batch, expect_retrain in ((stable, False), (shifted, True)):
        p = model.predict_proba(batch[FEATURES])[:, 1]
        report = drift_report(mv.reference, batch[FEATURES], p)
        decision = retrain_decision(report)
        assert decision.retrain is expect_retrain, decision.reasons
    assert "__score__" in set(report["feature"])


def test_performance_check():
    y = np.r_[np.zeros(900), np.ones(100)]
    p = np.r_[
        np.random.default_rng(0).uniform(0, 0.5, 900), np.random.default_rng(1).uniform(0.3, 1, 100)
    ]
    ok = performance_check(y, p, reference_auc=0.80, max_drop=0.03, min_labels=500)
    assert ok["evaluated"] and not ok["degraded"]
    bad = performance_check(y, np.random.default_rng(2).random(1000), 0.80, 0.03, 500)
    assert bad["degraded"]
    assert not performance_check(y[:100], p[:100], 0.8, 0.03, 500)["evaluated"]
