# Patient Report Demo: IHD-FUTURE-TTE-497

用途: 科研辅助/队列风险分层，不作为临床诊断依据。

## Snapshot

- Split: `test`; redacted patient index: `497`.
- Baseline age: `63.2` years; visible events: `8`.
- Hidden target: `diag:I25` (Ischemic heart disease (I20-I25)); first hidden event occurs after `7.0` days.
- Demo disease focus: `IHD` / Ischemic heart disease.

## Locked-Test Horizon Risk

The 5y/10y scores below are looked up from the locked-test final-context evaluator for the same redacted patient index. The demo hidden-event trajectory uses its own earlier cutpoint, so future-observed status is shown separately and should not be read as the locked-test label.

| Disease | Best token | Rank | 5y risk score | 10y risk score | Future observed in demo |
|---|---|---:|---:|---:|---|
| Ischemic heart disease (I20-I25) | `diag:I25` | 1 | 0.8182 | 0.8105 | yes (0.02y) |
| Myocardial infarction (I21-I22) | `diag:I21` | 2 | 0.1650 | 0.1748 | no |
| Atrial fibrillation and flutter (I48) | `diag:I48` | 3 | 0.0188 | 0.0163 | no |
| Heart failure (I50) | `diag:I50` | 4 | 0.0082 | 0.0078 | no |
| Hypertension (I10-I15) | `diag:I10` | 5 | 0.0039 | 0.0026 | no |

## Key Visible History

- 41.2y `proc:R249` (proc)
- 41.2y `diag:O14` (diag)
- 58.4y `diag:I21` (diag)
- 58.4y `proc:K751` (proc)
- 58.5y `diag:I20` (diag)
- 58.5y `proc:X998` (proc)
- 63.2y `proc:K634` (proc)

## Hidden Follow-Up Events For Demo Verification

- 63.2y `diag:I25` (diag)

## Model Evidence At Baseline

- Future `diag:I25` at 63.2y had baseline rank `1` within `diag` candidates.

## Interpretation Caveat

The report is a presentation demo built from held-out trajectory examples. Scores are model outputs for risk ranking and cohort stratification; they are not calibrated clinical probabilities for diagnosis or treatment decisions.
