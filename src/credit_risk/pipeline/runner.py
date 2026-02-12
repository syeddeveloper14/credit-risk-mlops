"""A small DAG runner with content-addressed caching.

A stage is skipped when its fingerprint (config parameters it depends on + hashes of its
input files + fingerprints of upstream stages) matches the last successful run and all of
its outputs still exist. Same idea as DVC or Make, kept dependency-free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Stage:
    name: str
    func: Callable[[Any], dict[str, Any]]
    deps: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: Callable[[Any], dict[str, Any]] = lambda ctx: {}
    cacheable: bool = True


@dataclass
class StageResult:
    name: str
    status: str  # "ran" or "cached"
    seconds: float
    summary: dict[str, Any] = field(default_factory=dict)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topological_order(stages: list[Stage]) -> list[Stage]:
    by_name = {s.name: s for s in stages}
    order: list[Stage] = []
    state: dict[str, str] = {}

    def visit(name: str) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise ValueError(f"Cycle detected at stage {name!r}")
        if name not in by_name:
            raise KeyError(f"Unknown stage {name!r}")
        state[name] = "visiting"
        for dep in by_name[name].deps:
            visit(dep)
        state[name] = "done"
        order.append(by_name[name])

