import pandas as pd
import pytest

from credit_risk.config import load_config
from credit_risk.data import simulate_applicants, stratified_split, to_canonical
from credit_risk.pipeline import run_pipeline


def make_config(root, **extra):
    overrides = {
        "paths.root": str(root),
        "registry.root": str(root / "model_registry"),
        "data.n_synthetic": 6000,
        "hpo.n_trials": 3,
        "hpo.cv_folds": 2,
        "models.candidates": ["scorecard", "logistic", "hgb"],
        "monitoring.prediction_log": str(root / "logs" / "predictions.jsonl"),
        # small test sets make slice metrics noisy; keep the gates but loosen them
        "promotion.min_auc": 0.70,
        "promotion.max_age_auc_gap": 0.30,
    }
    overrides.update(extra)
    return load_config("configs/pipeline.yaml", overrides)


@pytest.fixture(scope="session")
def applicants() -> pd.DataFrame:
    return to_canonical(simulate_applicants(8000, seed=3))


@pytest.fixture(scope="session")
def splits(applicants):
    return stratified_split(applicants)


@pytest.fixture(scope="session")
def pipeline_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("lifecycle")
    cfg = make_config(root)
    results = run_pipeline(cfg)
    return cfg, results
