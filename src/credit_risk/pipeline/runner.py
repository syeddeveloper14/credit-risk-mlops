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

    for s in stages:
        visit(s.name)
    return order


class PipelineRunner:
    def __init__(
        self, stages: list[Stage], context: Any, root: str | Path, cache_dir: str | Path
    ) -> None:
        self.stages = topological_order(stages)
        self.context = context
        self.root = Path(root)
        self.state_file = Path(cache_dir) / "state.json"
        self.state: dict[str, str] = (
            json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        )

    def _fingerprint(self, stage: Stage, upstream: dict[str, str]) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(stage.params(self.context), sort_keys=True, default=str).encode())
        for rel in stage.inputs:
            path = self.root / rel
            h.update(rel.encode())
            h.update(_file_hash(path).encode() if path.exists() else b"<missing>")
        for dep in stage.deps:
            h.update(upstream.get(dep, "").encode())
        return h.hexdigest()

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def run(self, until: str | None = None, force: bool | list[str] = False) -> list[StageResult]:
        stages = self.stages
        if until is not None:
            names = [s.name for s in stages]
            stages = stages[: names.index(until) + 1]
        forced = (
            set(force)
            if isinstance(force, list)
            else ({s.name for s in stages} if force else set())
        )

        results: list[StageResult] = []
        fingerprints: dict[str, str] = {}
        for stage in stages:
            # Upstream stages have already run (or been confirmed cached) at this point, so
            # input hashes reflect the files this stage will actually read.
            fp = self._fingerprint(stage, fingerprints)
            fingerprints[stage.name] = fp
            outputs_exist = all((self.root / o).exists() for o in stage.outputs)
            if (
                stage.cacheable
                and stage.name not in forced
                and self.state.get(stage.name) == fp
                and outputs_exist
            ):
                log.info("stage %-9s cached", stage.name)
                results.append(StageResult(stage.name, "cached", 0.0))
                continue
            log.info("stage %-9s running", stage.name)
            start = time.monotonic()
            summary = stage.func(self.context) or {}
            self.state[stage.name] = fp
            self._save_state()
            results.append(StageResult(stage.name, "ran", time.monotonic() - start, summary))
        return results
