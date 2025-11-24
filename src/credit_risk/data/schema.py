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
