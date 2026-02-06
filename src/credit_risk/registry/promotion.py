"""Champion / challenger promotion gates.

A challenger reaches Production only if every gate passes. Both models are scored on the
same, current test set so the comparison is like for like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config import PromotionConfig


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class PromotionDecision:
    promote: bool
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"promote": self.promote, "gates": [asdict(g) for g in self.gates]}

    def summary(self) -> str:
        lines = ["PROMOTE" if self.promote else "DO NOT PROMOTE"]
        lines += [f"  [{'x' if g.passed else ' '}] {g.name}: {g.detail}" for g in self.gates]
        return "\n".join(lines)


def decide_promotion(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    gates: PromotionConfig,
