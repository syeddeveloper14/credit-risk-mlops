# Data

Raw files go in `data/raw/` (git-ignored). The pipeline writes `data/interim/` and `data/processed/`.

## Give Me Some Credit (Kaggle)

1. Accept the competition rules at https://www.kaggle.com/c/GiveMeSomeCredit and download `cs-training.csv`, or:
   ```bash
   kaggle competitions download -c GiveMeSomeCredit -p data/raw && unzip data/raw/GiveMeSomeCredit.zip -d data/raw
   ```
2. In `configs/pipeline.yaml` set `data.source: file` and `data.path: data/raw/cs-training.csv`.
3. `credit-risk run`

`cs-test.csv` has no labels; it can be scored with `credit-risk score --input data/raw/cs-test.csv`.

## Columns

| Kaggle name | Canonical name | Notes |
|---|---|---|
| (unnamed first column) | `applicant_id` | Row id |
| SeriousDlqin2yrs | `serious_dlq_2yrs` | Target: 90+ days past due within 2 years |
| RevolvingUtilizationOfUnsecuredLines | `revolving_utilization` | Has extreme outliers (> 1,000) |
| age | `age` | |
| NumberOfTime30-59DaysPastDueNotWorse | `late_30_59` | 96 / 98 are codes, not counts |
| DebtRatio | `debt_ratio` | Dollar amount when income is missing |
| MonthlyIncome | `monthly_income` | ~20% missing |
| NumberOfOpenCreditLinesAndLoans | `open_credit_lines` | |
| NumberOfTimes90DaysLate | `late_90` | 96 / 98 codes |
| NumberRealEstateLoansOrLines | `real_estate_loans` | |
| NumberOfTime60-89DaysPastDueNotWorse | `late_60_89` | 96 / 98 codes |
| NumberOfDependents | `dependents` | ~2.6% missing |

## Synthetic batches

```bash
credit-risk simulate --n 5000 --out data/raw/batch.csv               # same population
credit-risk simulate --n 5000 --drift 1.2 --out data/raw/drifted.csv  # riskier, lower income
```
