# Patient Report Demo: CVD-FUTURE-TTE-356

用途: 科研辅助/队列风险分层，不作为临床诊断依据。

## Snapshot

- Split: `test`; redacted patient index: `356`.
- Baseline age: `67.9` years; visible events: `19`.
- Hidden target: `diag:I63` (Cerebrovascular disease (I60-I69)); first hidden event occurs after `11.0` days.
- Demo disease focus: `CVD` / Cerebrovascular disease.

## Locked-Test Horizon Risk

The 5y/10y scores below are looked up from the locked-test final-context evaluator for the same redacted patient index. The demo hidden-event trajectory uses its own earlier cutpoint, so future-observed status is shown separately and should not be read as the locked-test label.

| Disease | Best token | Rank | 5y risk score | 10y risk score | Future observed in demo |
|---|---|---:|---:|---:|---|
| Cerebrovascular disease (I60-I69) | `diag:I63` | 1 | 0.2967 | 0.2592 | yes (0.03y) |
| Atrial fibrillation and flutter (I48) | `diag:I48` | 2 | 0.0053 | 0.0059 | no |
| Ischemic heart disease (I20-I25) | `diag:I21` | 3 | 0.0010 | 0.0014 | no |
| Myocardial infarction (I21-I22) | `diag:I21` | 4 | 0.0006 | 0.0008 | no |
| Type 2 diabetes (E11) | `diag:E11` | 5 | 0.0008 | 0.0010 | no |

## Key Visible History

- 58.9y `proc:H202` (proc)
- 59.0y `proc:H231` (proc)
- 59.0y `diag:K62` (diag)
- 66.4y `proc:C601` (proc)
- 66.5y `diag:H40` (diag)
- 67.1y `diag:E83` (diag)
- 67.9y `proc:U051` (proc)
- 67.9y `proc:U543` (proc)

## Hidden Follow-Up Events For Demo Verification

- 67.9y `diag:I63` (diag)

## Model Evidence At Baseline

- Future `diag:I63` at 67.9y had baseline rank `1` within `diag` candidates.

## Interpretation Caveat

The report is a presentation demo built from held-out trajectory examples. Scores are model outputs for risk ranking and cohort stratification; they are not calibrated clinical probabilities for diagnosis or treatment decisions.
