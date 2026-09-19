# Patient Report Demo: CKD-FUTURE-TTE-346

用途: 科研辅助/队列风险分层，不作为临床诊断依据。

## Snapshot

- Split: `test`; redacted patient index: `346`.
- Baseline age: `61.7` years; visible events: `32`.
- Hidden target: `diag:N18` (Chronic kidney disease (N18)); first hidden event occurs after `6.0` days.
- Demo disease focus: `CKD` / Chronic kidney disease.

## Locked-Test Horizon Risk

The 5y/10y scores below are looked up from the locked-test final-context evaluator for the same redacted patient index. The demo hidden-event trajectory uses its own earlier cutpoint, so future-observed status is shown separately and should not be read as the locked-test label.

| Disease | Best token | Rank | 5y risk score | 10y risk score | Future observed in demo |
|---|---|---:|---:|---:|---|
| Chronic kidney disease (N18) | `diag:N18` | 1 | 0.0038 | 0.0033 | yes (0.02y) |
| Hypertension (I10-I15) | `diag:I12` | 2 | 0.0007 | 0.0007 | no |
| Type 2 diabetes (E11) | `diag:E11` | 3 | 0.0023 | 0.0007 | no |
| Atrial fibrillation and flutter (I48) | `diag:I48` | 4 | 0.0019 | 0.0015 | no |
| Heart failure (I50) | `diag:I50` | 5 | 0.0038 | 0.0043 | no |

## Key Visible History

- 60.9y `diag:I21` (diag)
- 60.9y `proc:K752` (proc)
- 61.4y `proc:W365` (proc)
- 61.4y `diag:D50` (diag)
- 61.6y `proc:X403` (proc)
- 61.6y `proc:K491` (proc)
- 61.6y `diag:I25` (diag)
- 61.7y `proc:X411` (proc)

## Hidden Follow-Up Events For Demo Verification

- 61.7y `diag:N18` (diag)
- 61.7y `proc:X412` (proc)
- 62.0y `proc:L932` (proc)
- 62.0y `proc:L742` (proc)
- 62.1y `proc:L747` (proc)
- 62.1y `proc:L743` (proc)

## Model Evidence At Baseline

- Future `diag:N18` at 61.7y had baseline rank `1` within `diag` candidates.
- Future `diag:I95` at 62.7y had baseline rank `58` within `diag` candidates.
- Future `diag:N17` at 62.7y had baseline rank `3` within `diag` candidates.
- Future `death:E14` at 63.0y had baseline rank `4` within `death` candidates.

## Interpretation Caveat

The report is a presentation demo built from held-out trajectory examples. Scores are model outputs for risk ranking and cohort stratification; they are not calibrated clinical probabilities for diagnosis or treatment decisions.
