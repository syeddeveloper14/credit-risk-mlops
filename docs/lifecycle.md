# Lifecycle design

## Stage contracts

Every stage in `pipeline/stages.py` is declared with:

| Field | Meaning |
|---|---|
| `deps` | Stages that must run (or be confirmed cached) first |
| `inputs` | Files the stage reads |
| `outputs` | Files the stage must produce |
| `params` | The slice of the config that affects the result |
| `cacheable` | Whether a matching fingerprint may skip the stage |

The fingerprint is a SHA-256 over the stage's config slice, the content hashes of its input files and its upstream fingerprints. A stage is skipped only when the fingerprint matches the last successful run and all outputs exist. Consequences:

* Changing `hpo.n_trials` re-runs `train` and everything after it; `ingest`, `validate` and `split` stay cached.
* Changing `split.seed` re-runs from `split`.
* If `train` is forced but produces a byte-identical model, `evaluate` and `register` are still skipped, because their input hashes did not change.
* `promote` is never cached: its outcome depends on the registry, which is outside the file graph.

`credit-risk run --force-stage train` re-runs one stage; `--force` re-runs all; `--until split` stops early.

## Reproducibility

* All randomness is seeded from the config (`data.seed`, `split.seed`, `hpo.seed`, `seed`).
* The registry stores, per version: the fitted pipeline, metrics, chosen hyperparameters, input signature, package version and the tracking run id.
* `reports/training.json` keeps the full candidate leaderboard, so it is clear why a family was selected.

## Data contract

`data/validate.py` separates blocking checks (missing columns, non-numeric features, duplicate ids, non-binary target, target rate outside the expected band, null rates above limits, negative values) from warnings (range violations within tolerance, sentinel codes). The real dataset triggers one warning by design: `debt_ratio` holds dollar amounts for ~20% of rows (those with missing income). The feature pipeline handles this explicitly instead of failing the run.

## Why candidates are tuned with cross-validation on train only

The validation split selects between families, and the test split is used once for the final report and the promotion gates. Tuning on validation as well would make the family comparison optimistic for whichever family had more hyperparameters. Optuna always evaluates the default parameters first, so a short search can only match or beat the defaults.

## Champion / challenger on the same data

When a challenger is registered, the current Production model is re-scored on the challenger's test set. Comparing against the champion's own historical metrics would mix two different test sets (and possibly two different populations if the data has drifted), which is the most common way promotion decisions go wrong.

## Monitoring loop

1. Every API prediction is appended to `logs/predictions.jsonl` (inputs, PD, decision, model version).
2. `credit-risk monitor` compares a batch with the reference profile saved at registration: PSI per input, PSI of the score, missing-value rates.
3. When labels have matured (serious delinquency within two years is a slow label), it also measures AUC against the registered test AUC.
4. `monitoring/policy.py` turns these signals into a decision with reasons; `.github/workflows/retrain.yml` runs this weekly and starts the pipeline when retraining is recommended. The new version still has to pass every promotion gate.
