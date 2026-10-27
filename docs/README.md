# Documentation

| File | Contents |
|---|---|
| `DATA.md` | How to obtain UK Biobank and MIMIC-IV access, which prepared inputs the code expects, and the environment variables that locate them |
| `REPRODUCE.md` | Reproduction path: environment, data preparation, training, evaluation and paired statistics |

## Where the reported numbers live

`results/aggregate/` contains the aggregate summaries behind the manuscript tables —
across-seed means, sample SDs, paired bootstrap intervals and per-stratum denominators.
Its `README.md` maps each file to the manuscript table it supports.

Two naming conventions coexist in this repository and are **not** interchangeable:

- **Track R / Track G** — the frozen protocol names used in `configs/` and `src/`.
  Track G is the locked-test multitype-generation protocol.
- **multitype-generation / diagnosis-only (N1, G1)** — the wording used in the manuscript.
  The N1 and G1 readouts are the diagnosis-only evaluations added later.

A manuscript claim should be traced to the aggregate file first, then to the protocol name,
then to the code path; the table in the repository README performs that mapping.
