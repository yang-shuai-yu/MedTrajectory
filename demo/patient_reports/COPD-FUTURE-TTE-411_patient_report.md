# Patient Report Demo: COPD-FUTURE-TTE-411

用途: 科研辅助/队列风险分层，不作为临床诊断依据。

## Snapshot

- Split: `test`; redacted patient index: `411`.
- Baseline age: `75.0` years; visible events: `16`.
- Hidden target: `diag:J44` (Chronic obstructive pulmonary disease (J44)); first hidden event occurs after `72.3` days.
- Demo disease focus: `COPD` / Chronic obstructive pulmonary disease.

## Locked-Test Horizon Risk

The 5y/10y scores below are looked up from the locked-test final-context evaluator for the same redacted patient index. The demo hidden-event trajectory uses its own earlier cutpoint, so future-observed status is shown separately and should not be read as the locked-test label.

| Disease | Best token | Rank | 5y risk score | 10y risk score | Future observed in demo |
|---|---|---:|---:|---:|---|
| Chronic obstructive pulmonary disease (J44) | `diag:J44` | 1 | 0.0056 | 0.0102 | yes (0.20y) |
| Heart failure (I50) | `diag:I50` | 2 | 0.0028 | 0.0019 | no |
| Atrial fibrillation and flutter (I48) | `diag:I48` | 3 | 0.0022 | 0.0019 | no |
| Ischemic heart disease (I20-I25) | `diag:I25` | 4 | 0.0004 | 0.0003 | no |
| Myocardial infarction (I21-I22) | `diag:I21` | 5 | 0.0002 | 0.0002 | no |

## Key Visible History

- 73.8y `proc:X998` (proc)
- 74.0y `proc:E852` (proc)
- 74.0y `proc:U201` (proc)
- 74.0y `diag:I47` (diag)
- 74.0y `diag:J96` (diag)
- 74.3y `diag:K85` (diag)
- 74.6y `diag:M54` (diag)
- 74.9y `diag:J22` (diag)

## Hidden Follow-Up Events For Demo Verification

- 75.2y `diag:J44` (diag)
- 75.2y `proc:U071` (proc)
- 75.7y `diag:F03` (diag)
- 76.2y `diag:F10` (diag)
- 76.2y `proc:U198` (proc)

## Model Evidence At Baseline

- Future `diag:J44` at 75.2y had baseline rank `2` within `diag` candidates.
- Future `diag:F03` at 75.7y had baseline rank `81` within `diag` candidates.
- Future `diag:F10` at 76.2y had baseline rank `17` within `diag` candidates.

## Interpretation Caveat

The report is a presentation demo built from held-out trajectory examples. Scores are model outputs for risk ranking and cohort stratification; they are not calibrated clinical probabilities for diagnosis or treatment decisions.
