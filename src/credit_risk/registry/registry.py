"""File-based model registry with stages (None -> Staging -> Production -> Archived).

Layout::

    model_registry/<model_name>/
        v1/  model.joblib  meta.json
        v2/  ...

Each version stores the fitted pipeline plus everything needed to serve and monitor it: the
input signature, evaluation metrics, training params, reference statistics for drift and
the tracking run that produced it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

STAGES = ("None", "Staging", "Production", "Archived")


@dataclass
class ModelVersion:
    name: str
    version: int
    stage: str
    created_at: str
    model_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    signature: list[str] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)


class ModelRegistry:
    def __init__(self, root: str | Path, model_name: str) -> None:
        self.dir = Path(root) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = model_name

    # ----------------------------------------------------------------- helpers
    def _version_dir(self, version: int) -> Path:
        return self.dir / f"v{version}"

    def _versions(self) -> list[int]:
        out = []
        for p in self.dir.iterdir():
            m = re.fullmatch(r"v(\d+)", p.name)
            if m and (p / "meta.json").exists():
                out.append(int(m.group(1)))
        return sorted(out)

    def _save_meta(self, mv: ModelVersion) -> None:
        (self._version_dir(mv.version) / "meta.json").write_text(
            json.dumps(asdict(mv), indent=2, default=str), encoding="utf-8"
        )

    # --------------------------------------------------------------- public API
    def register(
        self,
        model: Any,
        model_type: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        signature: list[str],
        reference: dict[str, Any] | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        version = (self._versions() or [0])[-1] + 1
        vdir = self._version_dir(version)
        vdir.mkdir(parents=True)
        joblib.dump(model, vdir / "model.joblib")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        mv = ModelVersion(
            name=self.name,
            version=version,
            stage="None",
            created_at=now,
            model_type=model_type,
            metrics=metrics,
            params=params,
            signature=signature,
            reference=reference or {},
            run_id=run_id,
            tags=tags or {},
            history=[{"time": now, "stage": "None", "note": "registered"}],
        )
        self._save_meta(mv)
        return mv

    def get(self, version: int) -> ModelVersion:
        path = self._version_dir(version) / "meta.json"
        if not path.exists():
            raise KeyError(f"{self.name} v{version} does not exist")
        return ModelVersion(**json.loads(path.read_text(encoding="utf-8")))

    def latest(self, stage: str | None = None) -> ModelVersion | None:
        for v in reversed(self._versions()):
            mv = self.get(v)
            if stage is None or mv.stage == stage:
                return mv
        return None

    def load_model(self, version: int) -> Any:
        return joblib.load(self._version_dir(version) / "model.joblib")

    def transition(
        self, version: int, stage: str, note: str = "", archive_existing: bool = True
    ) -> ModelVersion:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        mv = self.get(version)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if archive_existing and stage == "Production":
            for other in self._versions():
                o = self.get(other)
                if o.version != version and o.stage == "Production":
                    o.stage = "Archived"
                    o.history.append(
                        {"time": now, "stage": "Archived", "note": f"replaced by v{version}"}
                    )
                    self._save_meta(o)
        mv.stage = stage
        mv.history.append({"time": now, "stage": stage, "note": note})
        self._save_meta(mv)
        return mv

    def list_versions(self) -> pd.DataFrame:
        rows = []
        for v in self._versions():
            mv = self.get(v)
            rows.append(
                {
                    "version": mv.version,
                    "stage": mv.stage,
                    "model_type": mv.model_type,
                    "created_at": mv.created_at,
                    **{k: mv.metrics.get(k) for k in ("roc_auc", "ks", "ece")},
                }
            )
        return pd.DataFrame(rows)
