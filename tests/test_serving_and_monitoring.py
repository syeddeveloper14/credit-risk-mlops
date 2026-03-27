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
