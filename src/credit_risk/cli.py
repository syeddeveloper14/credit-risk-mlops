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
