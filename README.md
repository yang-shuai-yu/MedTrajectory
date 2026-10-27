# MedTrajectory

Reference implementation for **MedTrajectory**, a decoder-only autoregressive model
over multi-type longitudinal clinical event sequences. The model represents events
with absolute age and continuous relative time, and is evaluated on two tasks:

- **trajectory generation** — sampling future multi-type event sequences, and
- **fixed-horizon risk prediction** — 1-, 5- and 10-year disease risk heads on the
  frozen representation.

The repository also contains the MIMIC-IV cross-setting replication pipeline, which
retrains the same architecture on MIMIC-IV with a dataset-specific vocabulary.

> **Status.** This code accompanies a manuscript under revision. It is provided for
> review and reuse; see [Known limitations](#known-limitations) before drawing
> conclusions from any single entry point.

---

## Contents

| Path | What it contains |
|---|---|
| `src/semantic_delphi_ukb/` | UK Biobank pipeline: models, training, evaluation, the Track R / Track G contracts, statistical inference, baselines |
| `src/medtrajectory_mimic/` | MIMIC-IV pipeline: dataset construction, semantic initialisation, training drivers, generation and risk evaluation, paired analysis |
| `src/first_diagnosis_time_head/` | Independent first-diagnosis time head for the five models without operable native continuous time (Table III time column, Table S13e) |
| `scripts/` | Entry points and launchers (`.py` for analysis/evaluation, `.sh` for training queues) |
| `configs/` | Protocol definitions (clinical outcomes, horizons, landmarks, vocabulary) |
| `tests/` | Unit and contract tests |
| `demo/` | De-identified patient-level demonstration artefacts (`eid_redacted: true`) |
| `docs/` | Data access statement, reproduction guide and metric provenance |
| `results/aggregate/` | Aggregate numerical summaries behind the manuscript tables (across-seed means, paired intervals, denominators) |
| `supplementary/v14_submission/` | Manuscript and supplementary material as submitted (Tables I–IV; Tables S1–S12 and S13a–S13f) |
| `supplementary/` | The above, plus the earlier long-form supplementary material (S1–S33) retained for protocol detail |

---

## Manuscript artefacts

Where each reported number comes from. `T` = table in the manuscript, `S` = supplementary.

| Manuscript item | Aggregate source | Producing code |
|---|---|---|
| T I — event-space expansion | archived historical run | `src/semantic_delphi_ukb/` (Track R / Track G) |
| T II — next-diagnosis prediction (N1) | archived paired statistics | `src/semantic_delphi_ukb/`, `scripts/` |
| T III — untimed diagnosis generation + ten-year timing | archived paired statistics | `src/semantic_delphi_ukb/rollout_generation_benchmark.py` |
| T III time column, S13e — first-diagnosis time | archived timed evaluation | `src/first_diagnosis_time_head/` (five methods), native samplers (MT, Delphi, TCF) |
| T IV, S3 — corrected fixed-horizon risk | `results/aggregate/risk_corrected_configurations.csv`, `risk_noninferiority.json` | `src/semantic_delphi_ukb/` risk evaluation |
| S2 — paired generation differences | archived paired statistics | `scripts/` paired analysis |
| S5a–S5b — historical-gap strata | `results/aggregate/historical_gap_all_contrasts.csv` | `scripts/` paired analysis |
| S6–S8, Fig. 5 — MIMIC transfer | `results/aggregate/transfer_rollout_mean_sd.csv`, `transfer_rollout_paired_contrasts.csv` | `src/medtrajectory_mimic/` |
| S13a–S13f — diagnosis-only regime | archived paired statistics | `scripts/` paired analysis |
| Fig. 3, S7a–S7b — death tokens and mortality heads | `results/aggregate/death_blocking.csv`, `mortality_head_assessment.json` | `scripts/` death-mechanism analysis |
| Figs. S3–S6 — full-budget heatmaps | `results/aggregate/heatmap_values.csv` | `src/` + `scripts/` |

Two naming conventions are used interchangeably in the code base: **Track R / Track G**
(the frozen protocol names used in `configs/` and `src/`) and **multitype-generation /
diagnosis-only** (the wording used in the manuscript). Track G is the legacy locked-test
multitype-generation protocol; the N1 and G1 readouts reported in the manuscript are the
diagnosis-only evaluations.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Component-specific extras (MIMIC vocabulary build, Mamba/Med-BERT baseline arms,
Weights & Biases logging) are listed separately so that the core install stays light:

```bash
pip install -r requirements-optional.txt
```

For a **CPU-only PyTorch** environment:

```bash
pip install -r requirements_cpu_torch.txt   # pins torch to the CPU wheel index
```

Python >= 3.10 is assumed.

---

## Configuration: where the data lives

All locations are resolved from environment variables, so nothing is hard-wired to a
particular cluster. `src/semantic_delphi_ukb/paths.py` is the single source of truth
for Python code; `scripts/env.sh` mirrors it for shell launchers.

| Variable | Meaning | Default |
|---|---|---|
| `MEDTRAJECTORY_ROOT` | repository root | resolved from `paths.py` |
| `MEDTRAJECTORY_DATA_ROOT` | prepared datasets (contains `data/` and the paper-protocol directories) | `<root>/data` |
| `MEDTRAJECTORY_RUNS_ROOT` | run outputs | `<root>/results` |
| `MEDTRAJECTORY_MIMIC_ROOT` | MIMIC-IV project root (prepared sequences, vocab, runs) | `<root>/mimic-iv` |
| `MEDTRAJECTORY_EXTERNAL_ROOT` | third-party / reference-workspace artefacts | `<root>/external` |
| `MEDTRAJECTORY_PYTHON` | interpreter used by shell launchers | `python3` |
| `DASHSCOPE_API_KEY` | credential, only for regenerating MIMIC semantic embeddings | unset |

Check the resolved values at any time:

```bash
python -m semantic_delphi_ukb.paths      # requires src/ on PYTHONPATH
# or
source scripts/env.sh && echo "$MEDTRAJECTORY_DATA_ROOT"
```

Shell launchers source `scripts/env.sh` automatically.

---

## Data

**No participant-level data is distributed with this repository.** UK Biobank and
MIMIC-IV are both access-controlled and must be obtained by the user; see
[`docs/DATA.md`](docs/DATA.md) for access routes, the expected input layout, and the
terms that apply to third-party components (Qwen embeddings, Phecode maps,
PubMedBERT). This repository's licence covers the source code only.

---

## Running the pipeline

The pipeline consumes prepared, tokenised event sequences, not raw source tables.
The preparation entry points are listed in `docs/DATA.md` §2. Once a dataset profile
has been prepared under `MEDTRAJECTORY_DATA_ROOT`, the main drivers are:

| Stage | Entry point |
|---|---|
| UKB dataset preparation | `src/semantic_delphi_ukb/prepare_paper_protocol_dataset.py` |
| MedTrajectory pretraining | `src/semantic_delphi_ukb/train_car_rope_pretraining.py` |
| Risk-head post-training | `src/semantic_delphi_ukb/train_car_rope.py` |
| Fixed-horizon evaluation | `src/semantic_delphi_ukb/evaluate_track_r.py` |
| Landmark feature cache (baselines) | `scripts/build_track_r_landmark_features.py` |
| Track R head-to-head comparison | `src/semantic_delphi_ukb/compare_horizon_control_tasks.py` |
| Disease-level AUROC landscape | `scripts/evaluate_all_disease_lm.py` |
| ICD encoding-method comparison | `scripts/compare_icd_embeddings.py` |
| MIMIC dataset construction | `src/medtrajectory_mimic/mimic_build_dataset.py`, `mimic_build_visit.py`, `mimic_build_track_r.py` |
| MIMIC generation evaluation | `src/medtrajectory_mimic/mimic_generation_eval.py`, `mimic_generation_eval_trackr.py` |

Most scripts accept `--help`. Training queues are shell launchers under `scripts/`;
read the launcher before running it, as several are written for a GPU queue manager.

### Tests

```bash
pip install pytest
PYTHONPATH=src pytest tests
```

The tests cover the Track R / Track G contracts, the horizon-control metric
definitions, the calibration code, and the locked-test workflow, and do not require
access-controlled data.

---

## Reproducibility notes

- **Paths are parameterised, not hard-coded.** No absolute cluster path remains in the
  tree; if you find one, it is a bug.
- **Two evaluation protocols coexist.** The *paper protocol* uses a three-way
  participant split (training / validation / locked test); the older *explicit-split*
  experiments used an 80/20 split with no separate test partition. Do not compare
  numbers across the two.
- **Multiplicity.** Where a family of comparisons is reported, the frozen procedure
  and its family size are stated alongside the result. Some families cannot reach
  family-wise significance at the frozen replicate count; that is a property of the
  procedure and is reported as such.
- **Seeds.** Reported three-seed results average per-seed estimates. Where an
  ensemble is used, per-row prediction scores are averaged before metrics are
  computed — not the metrics themselves.

## Known limitations

1. **Data are not included**, so nothing here can be run end-to-end from a fresh
   clone without first obtaining UK Biobank and MIMIC-IV access.
2. **The path refactor is syntactic, not behavioural.** It was verified by compiling
   every Python file, parsing every shell script, and validating every JSON file, and
   by confirming that no cluster path remains. It has *not* been executed against
   real data on a fresh machine.
3. **A small number of one-off diagnostic scripts** from the development process are
   still present under `scripts/`; they are not part of the reproduction path.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the manuscript once available.

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
The licence covers the source code only; it does not extend to UK Biobank or
MIMIC-IV data, nor to third-party model weights and vocabularies.
