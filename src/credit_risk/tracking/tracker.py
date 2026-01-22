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

    def log_params(self, params: dict[str, Any]) -> None:
        flat: dict[str, Any] = {}
        _flatten("", params, flat)
        self._write("params.json", {**self._read("params.json"), **flat})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        current = self._read("metrics.json")
        current.update({k: float(v) for k, v in metrics.items()})
        self._write("metrics.json", current)
        with open(self._run_dir / "metrics_history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "time": _now(), **metrics}, default=float) + "\n")

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        target = self._run_dir / "artifacts" / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)

    def list_runs(self) -> pd.DataFrame:
        rows = []
        for run in sorted(self.root.iterdir()):
            if not (run / "meta.json").exists():
                continue
            meta = json.loads((run / "meta.json").read_text())
            metrics = (
                json.loads((run / "metrics.json").read_text())
                if (run / "metrics.json").exists()
                else {}
            )
            params = (
                json.loads((run / "params.json").read_text())
                if (run / "params.json").exists()
                else {}
            )
            rows.append(
                {
                    "run_id": meta["run_id"],
                    "name": meta["name"],
                    "status": meta["status"],
                    "start": meta["start"],
                    "_order": meta.get("start_ns", 0),
                    **{f"metric.{k}": v for k, v in metrics.items()},
                    **{f"param.{k}": v for k, v in params.items()},
                }
            )
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows).sort_values("_order").drop(columns="_order").reset_index(drop=True)
        )


class MLflowTracker:
    def __init__(self, uri: str, experiment: str) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'credit-risk-mlops[mlflow]'") from exc
        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    @property
    def run_id(self) -> str | None:
        run = self._mlflow.active_run()
        return run.info.run_id if run else None

    @contextmanager
    def start_run(self, name: str, tags: dict[str, str] | None = None):  # pragma: no cover
        with self._mlflow.start_run(run_name=name, tags=tags):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:  # pragma: no cover
        flat: dict[str, Any] = {}
        _flatten("", params, flat)
        self._mlflow.log_params(flat)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:  # pragma: no cover
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path) -> None:  # pragma: no cover
        self._mlflow.log_artifact(str(path))


def make_tracker(cfg: TrackingConfig, root: str | Path = ".") -> LocalTracker | MLflowTracker:
    if cfg.backend == "local":
        return LocalTracker(Path(root) / cfg.uri, cfg.experiment)
    if cfg.backend == "mlflow":
        return MLflowTracker(cfg.uri, cfg.experiment)
    raise ValueError(f"Unknown tracking backend {cfg.backend!r}")
