"""Canonical schema and data contract for the "Give Me Some Credit" dataset."""

from __future__ import annotations

from dataclasses import dataclass

TARGET = "serious_dlq_2yrs"
ID = "applicant_id"

RAW_TO_CANONICAL: dict[str, str] = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}

FEATURES: list[str] = [c for c in RAW_TO_CANONICAL.values() if c != TARGET]
PAST_DUE: list[str] = ["late_30_59", "late_60_89", "late_90"]
# In the public data, 96 and 98 in the past-due columns are codes, not counts.
SENTINEL_VALUES: tuple[int, ...] = (96, 98)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    minimum: float | None
    maximum: float | None
    nullable: bool
    description: str


CONTRACT: list[ColumnSpec] = [
    ColumnSpec(
        "revolving_utilization", 0, 1.5, False, "Balance on revolving lines / credit limits"
    ),
    ColumnSpec("age", 18, 110, False, "Age of borrower in years"),
    ColumnSpec("late_30_59", 0, 20, False, "Times 30-59 days past due (last 2 years)"),
    ColumnSpec("debt_ratio", 0, 5, False, "Monthly debt payments / monthly gross income"),
    ColumnSpec("monthly_income", 0, 100_000, True, "Monthly gross income"),
    ColumnSpec("open_credit_lines", 0, 60, False, "Open loans and lines of credit"),
    ColumnSpec("late_90", 0, 20, False, "Times 90+ days late"),
    ColumnSpec("real_estate_loans", 0, 30, False, "Mortgage and real estate loans"),
    ColumnSpec("late_60_89", 0, 20, False, "Times 60-89 days past due (last 2 years)"),
    ColumnSpec("dependents", 0, 20, True, "Number of dependents"),
]
