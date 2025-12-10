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
        Check(
            "required columns present",
            not missing,
            True,
            f"missing: {missing}" if missing else "ok",
        )
    )
    if missing:
        return report

    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    add(Check("feature columns numeric", not non_numeric, True, str(non_numeric) or "ok"))

    add(
        Check(
            "minimum row count",
            len(df) >= cfg.min_rows,
            True,
            f"{len(df)} rows (min {cfg.min_rows})",
        )
    )
    dup = int(df[ID].duplicated().sum())
    add(Check("unique applicant ids", dup == 0, True, f"{dup} duplicates"))

    if require_target:
        if TARGET not in df.columns:
            add(Check("target present", False, True, "target column missing"))
        else:
            values = set(df[TARGET].dropna().unique())
            add(Check("target is binary", values <= {0, 1}, True, f"values {sorted(values)}"))
            rate = float(df[TARGET].mean())
            lo, hi = cfg.target_rate_range
            add(
                Check(
                    "target rate in expected range",
                    lo <= rate <= hi,
                    True,
                    f"{rate:.4f} (expected {lo}-{hi})",
                )
            )

    for spec in CONTRACT:
        col = df[spec.name]
        null_rate = float(col.isna().mean())
        limit = cfg.max_null_rate.get(spec.name, 0.0 if not spec.nullable else 1.0)
        add(
            Check(
                f"{spec.name}: null rate",
                null_rate <= limit,
                True,
                f"{null_rate:.4f} (max {limit})",
            )
        )
        values = col.dropna()
        if spec.name in PAST_DUE:
            values = values[~values.isin(SENTINEL_VALUES)]
        out = ((values < spec.minimum) | (values > spec.maximum)).mean() if len(values) else 0.0
        add(
            Check(
                f"{spec.name}: range [{spec.minimum}, {spec.maximum}]",
                out <= cfg.max_out_of_range_rate,
                False,
                f"{out:.4%} out of range",
            )
        )
        if (values < 0).any():
            add(Check(f"{spec.name}: non-negative", False, True, "negative values found"))

    sentinel_rows = int(df[PAST_DUE].isin(SENTINEL_VALUES).any(axis=1).sum())
    add(Check("past-due sentinel codes", True, False, f"{sentinel_rows} rows with codes 96/98"))
    return report
