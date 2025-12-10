"""Data contract checks.

Each check is either *blocking* (the pipeline stops) or a *warning* (recorded in the report).
Range checks tolerate a small share of out-of-range values because the real data contains
known outliers that the feature pipeline handles explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..config import ValidationConfig
from .schema import CONTRACT, FEATURES, ID, PAST_DUE, SENTINEL_VALUES, TARGET


@dataclass
class Check:
    name: str
    passed: bool
    blocking: bool
    detail: str

    def __post_init__(self) -> None:
        self.passed = bool(self.passed)


@dataclass
class ValidationReport:
    n_rows: int
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class DataContractError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        lines = [f"- {c.name}: {c.detail}" for c in report.failures if c.blocking]
        super().__init__("Data contract violated:\n" + "\n".join(lines))
        self.report = report


def validate(
    df: pd.DataFrame, cfg: ValidationConfig | None = None, require_target: bool = True
) -> ValidationReport:
    cfg = cfg or ValidationConfig()
    report = ValidationReport(n_rows=len(df))
    add = report.checks.append

    missing = [c for c in [ID, *FEATURES] if c not in df.columns]
    add(
