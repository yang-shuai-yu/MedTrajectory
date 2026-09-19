# CKD / IHD / CVD / COPD Baseline vs TTE Demo

The four patient demos reuse the same test patient and target token as the existing modern-baseline demos, then rerun the same case with `MedTrajectory TTE multitask`.

## Patient-Level Demo

| Disease | Patient index | Target token | Baseline age | Follow-up | Modern diag rank | TTE diag rank | Modern group rank | TTE group rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CKD | 346 | `diag:N18` | 61.6838 | 1.2841 | 1 | 1 | 1 | 1 |
| IHD | 497 | `diag:I25` | 63.1814 | 0.0192 | 1 | 1 | 1 | 1 |
| CVD | 356 | `diag:I63` | 67.8686 | 0.0301 | 1 | 1 | 1 | 1 |
| COPD | 411 | `diag:J44` | 75.0027 | 1.2108 | 4 | 2 | 2 | 1 |

## Four-Disease Metric Context

| Disease | Positives | Modern AUC | TTE AUC | Delta AUC | Modern group Top10 | TTE group Top10 | Delta Top10 | Modern top-decile | TTE top-decile | TTE-head 5y AUC | TTE-head 10y AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 慢性肾病 (Chronic kidney disease) | 90 | 0.9426 | 0.9391 | -0.0036 | 0.7667 | 0.7667 | +0.0000 | 0.8444 | 0.8444 | 0.9027 | 0.8549 |
| 缺血性心脏病 (Ischemic heart disease) | 478 | 0.8244 | 0.8191 | -0.0054 | 0.6632 | 0.6778 | +0.0146 | 0.6172 | 0.6046 | 0.7364 | 0.6721 |
| 脑血管疾病 (Cerebrovascular disease) | 251 | 0.8280 | 0.8286 | +0.0006 | 0.5458 | 0.5498 | +0.0040 | 0.6056 | 0.6135 | 0.7569 | 0.6913 |
| 慢性阻塞性肺疾病 (Chronic obstructive pulmonary disease) | 17 | 0.9017 | 0.8993 | -0.0023 | 0.4706 | 0.4706 | +0.0000 | 0.8235 | 0.8235 | 0.8401 | 0.8021 |

## Output Files

- `tests/output/patient_future_demo_modern/CKD-FUTURE-MODERN-346_demo.md`
- `tests/output/patient_future_demo_modern/IHD-FUTURE-MODERN-497_demo.md`
- `tests/output/patient_future_demo_modern/CVD-FUTURE-MODERN-356_demo.md`
- `tests/output/patient_future_demo_modern/COPD-FUTURE-MODERN-411_demo.md`
- `tests/output/patient_future_demo_tte_multitask/CKD-FUTURE-TTE-346_demo.md`
- `tests/output/patient_future_demo_tte_multitask/IHD-FUTURE-TTE-497_demo.md`
- `tests/output/patient_future_demo_tte_multitask/CVD-FUTURE-TTE-356_demo.md`
- `tests/output/patient_future_demo_tte_multitask/COPD-FUTURE-TTE-411_demo.md`
- `tests/output/patient_future_demo_tte_multitask/ckd_ihd_cvd_copd_modern_vs_tte_demo.csv`
