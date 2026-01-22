"""Experiment tracking with one interface and two backends.

* ``LocalTracker`` writes runs as plain JSON files (no server, works in CI and tests).
* ``MLflowTracker`` forwards the same calls to MLflow when it is installed.

Pipeline code only talks to the interface, so switching backends is a config change.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ..config import TrackingConfig


class Tracker(Protocol):
    def start_run(self, name: str, tags: dict[str, str] | None = None): ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path) -> None: ...
    @property
    def run_id(self) -> str | None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


class LocalTracker:
    def __init__(self, root: str | Path = "mlruns_local", experiment: str = "default") -> None:
        self.root = Path(root) / experiment
        self.root.mkdir(parents=True, exist_ok=True)
        self._run_dir: Path | None = None
        self._run_id: str | None = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def _write(self, name: str, data: Any) -> None:
        assert self._run_dir is not None, "no active run"
        (self._run_dir / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _read(self, name: str) -> Any:
        assert self._run_dir is not None
        path = self._run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @contextmanager
    def start_run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[LocalTracker]:
        if self._run_dir is not None:
            raise RuntimeError("Nested runs are not supported")
        self._run_id = uuid.uuid4().hex[:12]
        self._run_dir = self.root / self._run_id
        (self._run_dir / "artifacts").mkdir(parents=True)
        meta = {
            "run_id": self._run_id,
            "name": name,
            "tags": tags or {},
            "start": _now(),
            "start_ns": time.time_ns(),  # for stable ordering of runs started in the same second
            "status": "RUNNING",
        }
        self._write("meta.json", meta)
        try:
            yield self
            meta["status"] = "FINISHED"
        except BaseException:
            meta["status"] = "FAILED"
            raise
        finally:
            meta["end"] = _now()
            self._write("meta.json", meta)
            self._run_dir, self._run_id = None, None

