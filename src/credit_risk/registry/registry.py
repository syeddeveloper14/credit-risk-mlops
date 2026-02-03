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
