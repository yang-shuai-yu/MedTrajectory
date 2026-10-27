# Aggregate numerical archive

Aggregate summaries behind the manuscript's numerical claims. All entries are
across-seed means, sample SDs, paired bootstrap intervals or per-stratum counts computed
from completed evaluation runs. No patient-level rows, identifiers or prediction dumps are
included.

| File | Manuscript / supplement target | Contents |
|---|---|---|
| `risk_corrected_configurations.csv` | Tables IV, S3 | AUROC / AUPRC / Brier / ECE per configuration, for both the original and the eligibility-corrected test population (`population` column) |
| `risk_noninferiority.json` | Section V.C, Table S3 note | Per-seed Relative−Absolute AUROC differences, the 10,000-draw paired bootstrap interval, patient and row counts, and the −0.005 noninferiority decision |
| `death_blocking.csv` | Section V.D, Figure 3 | Validation mortality-token blocking: original vs blocked Jaccard and count MAE, paired deltas and 95% intervals |
| `mortality_head_assessment.json` | Section V.D, Table S7a–S7b | Frozen-backbone mortality heads at 1/5/10 years: raw and calibrated AUROC, AUPRC, Brier, ECE, mean probability, cases and N |
| `historical_gap_all_contrasts.csv` | Tables S5a–S5b | Historical-gap quartile contrasts for all eight generation endpoints, with intervals, denominators and seed SD |
| `transfer_rollout_mean_sd.csv` | Tables S6–S7, Figure 5 | MIMIC-IV free-generation rollout metrics per family × initialization: Jaccard, recall, count MAE, first-record time MAE, edit distance, unknown / empty / cap fractions |
| `transfer_rollout_paired_contrasts.csv` | Table S8 | Paired Jaccard contrasts for initialization gain, between-family differences and gain differences |
| `heatmap_values.csv` | Figures S3–S6 | Full-budget and 10k-update next-record and generation heatmaps: Hit@10, cross-entropy, waiting-time NLL and their sum |

Column conventions:

- `seed_sd` is the sample SD (`ddof = 1`) across seeds 42–44, not a standard error.
- Bootstrap intervals are pointwise, conditional on the three fitted seeds, and unadjusted
  unless a Holm or Bonferroni family is stated in the manuscript.
- Denominators are reported per metric; ``patients`` differs between endpoints because a
  participant enters a paired contrast only when both arms have a finite value.

These files are the source of the numbers in `supplementary/v14_submission/`. The
long-form supplementary document under `supplementary/` was written against an earlier
experiment version and may quote superseded values.
