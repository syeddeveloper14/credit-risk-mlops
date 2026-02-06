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
) -> PromotionDecision:
    results: list[GateResult] = []

    def gate(name: str, passed: bool, detail: str) -> None:
        results.append(GateResult(name, bool(passed), detail))

    auc = challenger["roc_auc"]
    gate("minimum AUC", auc >= gates.min_auc, f"{auc:.4f} >= {gates.min_auc}")
    gate(
        "calibration (ECE)",
        challenger["ece"] <= gates.max_ece,
        f"{challenger['ece']:.4f} <= {gates.max_ece}",
    )
    gate(
        "score stability (train vs test PSI)",
        challenger["score_psi"] <= gates.max_score_psi,
        f"{challenger['score_psi']:.4f} <= {gates.max_score_psi}",
    )
    gate(
        "age-band AUC gap",
        challenger["age_auc_gap"] <= gates.max_age_auc_gap,
        f"{challenger['age_auc_gap']:.4f} <= {gates.max_age_auc_gap}",
    )
    gate(
        "approval-rate ratio across age bands",
        challenger["approval_ratio"] >= gates.min_approval_ratio,
        f"{challenger['approval_ratio']:.3f} >= {gates.min_approval_ratio}",
    )
    if champion is None:
        gate("beats champion", True, "no champion in Production")
    else:
        delta = auc - champion["roc_auc"]
        gate(
            "beats champion",
            delta >= gates.min_auc_improvement,
            f"AUC {auc:.4f} vs {champion['roc_auc']:.4f} (delta {delta:+.4f}, need >= {gates.min_auc_improvement})",
        )
    return PromotionDecision(promote=all(g.passed for g in results), gates=results)
