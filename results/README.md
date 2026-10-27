# Results

Only summary-level artefacts are published here. Patient-level predictions, per-participant
rows and model checkpoints are deliberately excluded (see `.gitignore`).

| Path | Contents |
|---|---|
| `aggregate/` | Aggregate numerical summaries behind the manuscript tables and supplementary tables |

`aggregate/` holds the values that the manuscript reports: three-seed means and sample SDs,
paired bootstrap contrasts, strata counts and denominators. Everything in it is derived from
the completed evaluation runs; nothing is regenerated at install time and none of it can be
traced back to an individual participant of UK Biobank or MIMIC-IV.

Reproducing the reported numbers end-to-end still requires credentialed access to the two
source databases (see `../docs/DATA.md`).
