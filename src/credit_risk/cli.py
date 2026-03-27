"""``credit-risk`` command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import load_config

STAGE_NAMES = ["ingest", "validate", "split", "train", "evaluate", "register", "promote"]


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    cfg = load_config(args.config)
    force = args.force_stage or args.force
    results = run_pipeline(cfg, until=args.until, force=force)
    for r in results:
        print(f"{r.name:<9} {r.status:<6} {r.seconds:6.1f}s  {json.dumps(r.summary, default=str)}")
    return 0


def _cmd_registry(args: argparse.Namespace) -> int:
    from .registry import ModelRegistry

    cfg = load_config(args.config)
    registry = ModelRegistry(cfg.registry.root, cfg.registry.model_name)
    if args.action == "list":
        table = registry.list_versions()
        print(table.to_string(index=False) if len(table) else "registry is empty")
    elif args.action == "transition":
        if args.version is None or args.stage is None:
            print("--version and --stage are required", file=sys.stderr)
            return 2
        mv = registry.transition(args.version, args.stage, note="manual")
        print(f"{mv.name} v{mv.version} -> {mv.stage}")
    elif args.action == "rollback":
        current = registry.latest("Production")
        previous = next(
            (
                registry.get(v)
                for v in range((current.version if current else 1) - 1, 0, -1)
                if registry.get(v).stage == "Archived"
            ),
            None,
        )
        if previous is None:
            print("no archived version to roll back to", file=sys.stderr)
            return 1
        registry.transition(previous.version, "Production", note="rollback")
        print(f"rolled back to v{previous.version}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .data.ingest import to_canonical
    from .registry import ModelRegistry
    from .serving import CreditRiskPredictor

    cfg = load_config(args.config)
    predictor = CreditRiskPredictor.from_registry(
        ModelRegistry(cfg.registry.root, cfg.registry.model_name), cfg.scoring
    )
    df = to_canonical(pd.read_csv(args.input))
    out = predictor.predict(df)
    out["reasons"] = out["reasons"].map(lambda r: ";".join(r))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"scored {len(out):,} applicants with {predictor.version} -> {args.output}")
    print(out["decision"].value_counts(normalize=True).round(3).to_string())
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from .data.ingest import to_canonical
    from .data.schema import FEATURES, TARGET
    from .monitoring import drift_report, performance_check, retrain_decision
    from .registry import ModelRegistry

    cfg = load_config(args.config)
    registry = ModelRegistry(cfg.registry.root, cfg.registry.model_name)
    mv = registry.latest("Production")
    if mv is None:
        print("no Production model", file=sys.stderr)
        return 1
    model = registry.load_model(mv.version)
    df = to_canonical(pd.read_csv(args.input))
    p = model.predict_proba(df[FEATURES])[:, 1]
    m = cfg.monitoring
    drift = drift_report(mv.reference, df[FEATURES], p, m.psi_warn, m.psi_alert)
    perf = None
    if TARGET in df.columns and df[TARGET].notna().any():
        perf = performance_check(
            df[TARGET], p, mv.metrics["roc_auc"], m.max_auc_drop, m.min_labels_for_performance
        )
    decision = retrain_decision(drift, perf)
    print(drift.to_string(index=False))
    if perf:
        print(json.dumps(perf, indent=2))
    print(
        ("RETRAIN RECOMMENDED: " + "; ".join(decision.reasons))
        if decision.retrain
        else "No action needed"
    )
    return 3 if decision.retrain and args.exit_code else 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .data import simulate_applicants

    df = simulate_applicants(args.n, seed=args.seed, drift=args.drift)
    if args.no_labels:
        df = df.drop(columns=["SeriousDlqin2yrs"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"wrote {len(df):,} applicants (drift={args.drift}) to {args.out}")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from .tracking import LocalTracker

    cfg = load_config(args.config)
    tracker = LocalTracker(Path(cfg.paths.root) / cfg.tracking.uri, cfg.tracking.experiment)
    runs = tracker.list_runs()
    cols = [
        c
        for c in runs.columns
        if c in ("run_id", "name", "status", "start")
        or c.startswith("metric.valid_roc")
        or c == "metric.cv_auc"
    ]
    print(runs[cols].to_string(index=False) if len(runs) else "no runs yet")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serving.api import create_app

    uvicorn.run(create_app(cfg=load_config(args.config)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-risk")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--config", default="configs/pipeline.yaml")
        return p

    p = with_config(
        sub.add_parser("run", help="run the lifecycle pipeline (cached stages are skipped)")
    )
    p.add_argument("--until", choices=STAGE_NAMES)
    p.add_argument("--force", action="store_true", help="re-run every stage")
    p.add_argument("--force-stage", action="append", choices=STAGE_NAMES, help="re-run one stage")
    p.set_defaults(func=_cmd_run)

    p = with_config(sub.add_parser("registry", help="inspect or change model stages"))
    p.add_argument("action", choices=["list", "transition", "rollback"])
    p.add_argument("--version", type=int)
    p.add_argument("--stage", choices=["None", "Staging", "Production", "Archived"])
    p.set_defaults(func=_cmd_registry)

    p = with_config(sub.add_parser("score", help="batch-score a CSV with the Production model"))
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="reports/scored.csv")
    p.set_defaults(func=_cmd_score)

    p = with_config(sub.add_parser("monitor", help="drift + performance check on a new batch"))
    p.add_argument("--input", required=True)
    p.add_argument("--exit-code", action="store_true", help="exit 3 when retraining is recommended")
    p.set_defaults(func=_cmd_monitor)

    p = sub.add_parser("simulate", help="write a synthetic applicant batch (optionally drifted)")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--drift", type=float, default=0.0)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--out", default="data/raw/batch.csv")
    p.set_defaults(func=_cmd_simulate)

    p = with_config(sub.add_parser("runs", help="list tracked experiment runs"))
    p.set_defaults(func=_cmd_runs)

    p = with_config(sub.add_parser("serve", help="start the prediction API"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
