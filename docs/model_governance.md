# Model governance

Credit models make decisions about people, so this repository treats explainability, stability and disparity checks as release criteria rather than optional analysis.

## Promotion gates

| Gate | Why it exists |
|---|---|
| Minimum AUC | Floor on discrimination; below it the model is not worth the operational risk |
| Beats champion by a margin | Avoids churning versions for noise-level gains; every release has a cost (monitoring reset, documentation, possible policy change) |
| Expected calibration error | Decisions use PD cut-offs and pricing uses PD directly; a well-ranked but miscalibrated model approves the wrong volume |
| Train-vs-test score PSI | Flags overfitting or a population mismatch between training and evaluation data |
| AUC gap across age bands | The model should rank risk about equally well for every age group |
| Approval-rate ratio across age bands | Adverse-impact screen in the spirit of the four-fifths rule |

ECE is computed over equal-count bins: with a ~6% default rate, equal-width bins above 0.3 are nearly empty and hide miscalibration where most applicants sit.

## Fairness slices: what they do and do not show

Age is a model input in this dataset and also a characteristic protected in many jurisdictions. In the US, ECOA / Regulation B permits age in an empirically derived, statistically sound scoring system but forbids assigning a negative factor to applicants aged 62 or older. The age-band slices make the model's behaviour across age groups visible; they are not a legal review. Before production use:

* confirm with compliance which characteristics may be inputs and which may appear in adverse-action reasons;
* check the scorecard points table so that the 62+ band never receives fewer points than younger bands;
* consider removing age and measuring the performance cost.

Approval-rate differences can come from genuine differences in observed risk. The gate forces that discussion; it does not settle it.

## Reason codes

Declined and referred applicants receive up to four reasons.

* **Scorecard**: for each characteristic, the gap between the applicant's points and the best attainable points for that characteristic. The largest gaps are the reasons. This is the standard way adverse-action reasons are produced from points-based scorecards.
* **Other models**: each raw input is replaced by the median of good applicants from training and the change in PD is measured. Inputs whose replacement lowers PD the most are the reasons.

Reasons are not returned for approved applicants. Human-readable text lives in `serving/predictor.py::REASON_TEXT` and should be reviewed by compliance before use in customer communications.

## Score scaling

```
score  = offset + factor · ln(odds_good)
factor = PDO / ln 2
offset = base_score − factor · ln(base_odds)
```

Defaults: 600 points at 50:1 good:bad odds, 20 points to double the odds. The decision bands are defined on PD (`approve_below_pd`, `decline_above_pd`) so they stay meaningful if the scaling changes.

## Known limitations

* The public dataset has no application dates, so splits are stratified rather than time-based. With dated data, use out-of-time validation.
* Labels take two years to mature; performance monitoring will lag drift monitoring by design.
* Reject inference is out of scope: the model only learns from approved, booked accounts.
