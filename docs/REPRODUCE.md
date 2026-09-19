# Reproduction guide

This guide maps each reported analysis to the commands that produce it. Read
[Scope and status](#scope-and-status) first: it states exactly what has been
verified and what has not.

---

## Scope and status

**What is verified**

- Every Python file compiles, every shell script parses (`bash -n`), and every JSON
  file is valid. Counts at the time of writing: 292 `.py`, 107 `.sh`, 31 `.json`.
- No absolute cluster path or internal username remains anywhere in the tree.
- The public entry points accept the flags quoted below 鈥?these were extracted from
  the scripts' own `argparse` definitions, not written from memory.

**What is NOT verified**

- **No command in this guide has been executed end-to-end on a fresh machine.** The
  development runs happened on a specific allocation; the guide states the intended
  sequence, not a tested one.
- The path-parameterisation refactor is **syntactic**. If a stage fails immediately
  with a missing-file error, check `python -m semantic_delphi_ukb.paths` first.
- Several launchers under `scripts/` were written for a GPU queue manager (GPU index,
  `tmux` session names, concurrency). Review them before running.

---

## 1. Environment

```bash
git clone https://github.com/yang-shuai-yu/MedTrajectory.git MedTrajectory && cd MedTrajectory
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt                # only for the components you use
pip install pytest                                      # to run the tests
```

Python >= 3.10. For CPU-only PyTorch use `requirements_cpu_torch.txt`.

Point the code at your data (see [`DATA.md`](DATA.md) for how to obtain it):

```bash
export MEDTRAJECTORY_ROOT="$(pwd)"
export MEDTRAJECTORY_DATA_ROOT=/path/to/prepared/datasets
export MEDTRAJECTORY_RUNS_ROOT=/path/to/run/outputs
export MEDTRAJECTORY_MIMIC_ROOT=/path/to/mimic-iv            # MIMIC stages only
export MEDTRAJECTORY_PYTHON="$(which python)"                # used by shell launchers

python -m semantic_delphi_ukb.paths                          # with src/ on PYTHONPATH
```

Shell launchers pick these up through `scripts/env.sh`; they do not need the
`export` lines above if you place `data/` and `results/` inside the repository.

Sanity check before anything else:

```bash
PYTHONPATH=src pytest tests -q
```

The tests cover the Track R / Track G contracts, the horizon-control metric
definitions, the calibration code and the locked-test workflow, and need no
access-controlled data.

---

## 2. Two protocols coexist 鈥?do not mix them

| | Paper protocol | Legacy explicit split |
|---|---|---|
| Split | training / validation / **locked test** | 80 / 20, **no test** |
| Participants | 397,049 / 99,263 / 6,099 | 401,928 / 100,483 |
| Launcher | `scripts/run_paper_protocol_experiment.sh` | `scripts/oneoff/run_medtrajectory_*_train.sh` |
| Used for | the reported results | development only |

Numbers from the two are **not comparable**. The legacy launchers were moved to
`scripts/oneoff/` for exactly this reason.

---

## 3. Data preparation

### 3.1 UK Biobank

```bash
# rebuild the paper-protocol profiles from the canonical records
python -m semantic_delphi_ukb.prepare_paper_protocol_dataset \
  --canonical-jsonl "$MEDTRAJECTORY_DATA_ROOT/canonical/patient_records_core.jsonl" \
  --static-csv      "$MEDTRAJECTORY_DATA_ROOT/model_input/static_features_v1.csv" \
  --source-vocab-csv "$MEDTRAJECTORY_DATA_ROOT/vocab/dynamic_token_vocab.csv" \
  --current-data-dir "$MEDTRAJECTORY_DATA_ROOT/ukb_semantic_multitype_explicit_split" \
  --output-dir       "$MEDTRAJECTORY_DATA_ROOT/paper_protocol_v1" \
  --seed 1337
```

`--profiles` selects which dataset profiles to emit (`diagnosis_death`,
`multitype`, `reduced_multitype`). The script writes, per profile, the
`*.bin` sequence files, `*_patient_index.csv`, static features and the semantic
embedding matrix, plus a `prepare_manifest.json` recording the split method and
counts.

> The split method recorded in that manifest is what determines which protocol you
> are in. Check it before quoting any number.

### 3.2 MIMIC-IV

```bash
cd "$MEDTRAJECTORY_MIMIC_ROOT"
python "$MEDTRAJECTORY_ROOT/src/medtrajectory_mimic/mimic_build_dataset.py"   # see --help
python "$MEDTRAJECTORY_ROOT/src/medtrajectory_mimic/mimic_build_visit.py"     # visit-level
python "$MEDTRAJECTORY_ROOT/src/medtrajectory_mimic/mimic_build_track_r.py"   # Track-R prefix
```

The MIMIC builders read the raw `mimic-iv-3.1/` tables and the PheWAS maps, and
write the same `.bin` contract as the UKB side. Semantic token initialisation
requires a DashScope credential:

```bash
export DASHSCOPE_API_KEY=...            # only for regenerating embeddings
python "$MEDTRAJECTORY_ROOT/src/medtrajectory_mimic/mimic_qwen_embed.py"
```

Precomputed embedding matrices are **not** distributed (they derive from the
access-controlled vocabulary); regenerate them, or supply your own 64-d matrix in
the documented row order (row 0 = padding zeros, row 1 = no-event zeros, rows 2+ =
dynamic tokens in vocabulary order).

---

## 4. Freeze, train, evaluate

### 4.1 Freeze the locked test before any evaluation

```bash
# 1. audit the split and emit the split-audit artefact
python scripts/audit_paper_protocol_splits.py --help

# 2. build the immutable input manifest
python scripts/build_paper_protocol_freeze_manifest.py \
  --split-audit  <split_audit.json> \
  --data-dir     "$MEDTRAJECTORY_DATA_ROOT/paper_protocol_v1/multitype" \
  --diseases-yaml configs/.../selected_diseases.yaml \
  --checkpoint   <best_val_loss.pt> \
  --protocol-json <protocol.json> \
  --test-eids-csv paper_protocol_v1/locked_test_eids.csv \
  --out          results/paper_protocol_v1/locked_test_<date>/freeze_manifest.json

# 3. re-hash every frozen input before anything reads test data
python scripts/verify_paper_protocol_freeze_manifest.py \
  --freeze-manifest .../freeze_manifest.json --out .../freeze_verification.json
```

**Always run step 3 before an evaluation that touches the test partition.** It is
the guard that makes the "one-shot locked test" claim meaningful.

### 4.2 Train

The paper-protocol profiles (P0鈥揚4) run pretraining, risk-head post-training, and
both evaluations as one sequence:

```bash
bash scripts/run_paper_protocol_experiment.sh P3 42        # key, seed
```

Behind it, the individual stages are:

```bash
python -m semantic_delphi_ukb.train_paper_protocol_pretraining \
  --experiment-config configs/paper_protocol_v1/P3.json \
  --data-dir "$MEDTRAJECTORY_DATA_ROOT/paper_protocol_v1/multitype" \
  --seed 42 --run-dir <run>/pretraining --device cuda

python -m semantic_delphi_ukb.train_paper_protocol_risk_heads \
  --experiment-config configs/paper_protocol_v1/P3.json \
  --data-dir "$MEDTRAJECTORY_DATA_ROOT/paper_protocol_v1/multitype" \
  --seed 42 --init-from-ckpt <run>/pretraining/checkpoints/best_val_loss.pt \
  --run-dir <run>/risk --device cuda
```

Track R v2.2 (the relative-time / risk study) is driven by a lane runner:

```bash
python scripts/run_track_r_v2_2.py --lane all --device cuda        # renders; add --execute to run
python scripts/run_track_r_v2_2.py --lane all --device cuda --execute
```

### 4.3 Evaluate

```bash
# fixed-horizon risk evaluation for one model
python -m semantic_delphi_ukb.evaluate_track_r \
  --family carope --model-name A2 --checkpoint <ckpt> \
  --protocol configs/paper_protocol_v1/TRACK_R_v2_1.json \
  --data-dir "$MEDTRAJECTORY_DATA_ROOT/paper_protocol_v1/multitype" \
  --split val --landmark-manifest <landmarks.json> \
  --out-dir <run>/validation --include-static-prefix true --device cuda

# generation evaluation (Track G) and the frozen landmark manifest
python -m semantic_delphi_ukb.evaluate_track_g_generation --help
python scripts/build_shared_test_landmarks.py --help
```

---

## 5. Analysis by analysis

Each row names the paper element, the producing command, and the key flags. Flags
shown are the real ones; consult `--help` for the full set.

### 5.1 Trajectory generation (Diagnosis Jaccard, Hit@10, time/count MAE)

| Step | Command |
|---|---|
| Rollouts and metrics | `python -m semantic_delphi_ukb.evaluate_track_g_generation --help` |
| Sampling-quality selector | `python -m semantic_delphi_ukb.select_track_g_sampler --help` |
| Finalise the Track G cohort | `python -m semantic_delphi_ukb.finalize_track_g --help` |

Rollout parameters (from the MIMIC evaluator, mirrored on the UKB side):
`--num-rollouts 20`, `--max-new-tokens 30`, `--followup-years 10`,
`--baseline-fraction 0.65`, `--temperature 0.8`, `--top-p 0.9`,
`--min-history-events 8`, `--min-future-events 3`.

### 5.2 Fixed-horizon risk comparison (Track R)

| Step | Command |
|---|---|
| Landmark feature cache for baselines | `python scripts/build_track_r_landmark_features.py --protocol ... --data-dir ... --split train --landmark-manifest ... --output .../train.npz` |
| Baseline arms (logistic / Cox / MDRMF) | `python -m semantic_delphi_ukb.track_r_baselines --model mdrmf-clinical --train-features ... --val-features ... --eval-features ... --output-dir ...` |
| Nine-model head-to-head | `python -m semantic_delphi_ukb.compare_horizon_control_tasks --input <name=path> ... --out-dir ... --bootstrap 1000 --seed 42 --calibration-bins 10 --memory-efficient` |
| Three-seed ensemble rows | `python scripts/aggregate_seed_rows.py --input a --input b --input c --output ... --model <name>` |
| Locked-test risk aggregation | `python scripts/aggregate_test_risk.py --help` |
| Formal paired bootstrap | `python scripts/aggregate_test_risk_bootstrap.py --help` |

`compare_horizon_control_tasks` is the module that computes the macro-AUROC family,
the bootstrap intervals and the Holm adjustment. Its behaviour on ties, empty
strata and shared patient resamples is covered by `tests/test_horizon_control_tasks.py`.

**Performance note.** The shipped bootstrap is a single-process loop; on a
validation-sized population (鈮?.5M rows) 10,000 replicates take on the order of
hours. `scripts/parallel_bootstrap.py` is an execution-equivalent parallel
replacement 鈥?it reproduces the shipped function's output **bit-for-bit** on the
test data and was validated by reproducing a published interval exactly. Its
docstring states the equivalence evidence and its preconditions; use it when you
need the frozen number faster.

### 5.3 Disease-level AUROC landscape

```bash
python scripts/evaluate_all_disease_lm.py \
  --checkpoint <risk-posttrained-ckpt> \
  --protocol configs/track_r_v2_2/TRACK_R_v2_2.json \
  --data-dir "$MEDTRAJECTORY_DATA_ROOT/track_r_v2_1/multitype_static_prefix" \
  --split val --vocab-csv <.../vocab/dynamic_token_vocab.csv> \
  --out-dir results/all_disease_lm --device cuda --min-cases 1
```
Outputs `all_disease_lm_auc.csv` (one row per diagnosis token) and `summary.json`.

The reported table uses the subset with **>= 20 future cases**; pooled AUROC uses
all scored participants, while the age鈥搒ex stratified value is a separate column.
Confidence intervals are DeLong-based; the CSV reports them, and any bounded
(logit-delta) variant must be applied to the whole set rather than to the
out-of-range entries alone.

Figures:

```bash
python scripts/plot_disease_landscape.py --help          # Figure 5A / 5B
python scripts/plot_all_disease_landscape.py --help      # full landscape
```

### 5.4 Extended-panel and horizon-specific risk tables

```bash
python scripts/summarize_disease_level_risk.py \
  --rows <evaluate_track_r output>/rows.json.gz \
  --diseases-yaml configs/.../expanded_disease_panel_ukb_icd10.yaml \
  --out-dir <...>/disease_level --bins 10 --bootstrap 1000 --workers 8
```
Writes `disease_horizon_risk.csv`, `disease_horizon_sex_risk.csv`,
`disease_horizon_age_risk.csv`, `disease_macro_risk.csv` and `summary.json`.

### 5.5 ICD encoding-method comparison (Supplementary Figure S2)

```bash
export DASHSCOPE_API_KEY=...        # only needed for the 'qwen' method
python scripts/compare_icd_embeddings.py --methods hierarchy gram qwen pubmedbert \
  --icd-dir <icd10 ontology dir> --out outputs/icd_embedding_compare
python scripts/plot_supp_figS2_icd_umap.py --help
```

This is a **method-comparison study**; its 64-d matrices are **not** the embeddings
the model consumes. See `DATA.md` 搂3.

### 5.6 MIMIC cross-setting replication

```bash
python src/medtrajectory_mimic/mimic_generation_eval.py \
  --ckpt <mimic ckpt> --data-dir "$MEDTRAJECTORY_MIMIC_ROOT/multitype" --split test \
  --out-dir <...>/eval --num-rollouts 20 --min-history-events 8 --min-future-events 3
python src/medtrajectory_mimic/mimic_generation_eval_trackr.py --help
python src/medtrajectory_mimic/mimic_paired_3seed.py --help       # paired 3-seed analysis
python src/medtrajectory_mimic/verify_eval_cohorts.py --help      # eid-level pairing gate
```

`verify_eval_cohorts.py` refuses to proceed when the evaluated patient sets differ
between arms. **Run it before trusting a paired comparison** 鈥?omitting the cohort
file silently changes the evaluated population.

### 5.7 Supplementary figures

```bash
python scripts/plot_death_mechanism.py --help                    # Figure 6
python scripts/plot_relative_time_forest.py --help               # Figure 4
python scripts/plot_supp_figS1_death_calibration.py --help       # Figure S1
python scripts/plot_supp_figS2_icd_umap.py --help                # Figure S2
```

---

## 6. Expected outputs

Per run, the layout is:

```text
<run>/
鈹溾攢鈹€ pretraining/checkpoints/       last.pt, best_val_loss.pt
鈹溾攢鈹€ risk/checkpoints/              last.pt, best_val_loss.pt
鈹溾攢鈹€ validation/                    rows.json.gz, summary.json
鈹溾攢鈹€ medical_auc/                   summary.json
鈹溾攢鈹€ longitudinal_auc/              summary.json
鈹溾攢鈹€ disease_level/                 per-disease and per-horizon tables  (if run)
鈹斺攢鈹€ resource_decision.txt          resolved data dir, device, GPU id
```

`rows.json.gz` is the patient-level record consumed by every downstream
aggregation and paired comparison.

---

## 7. When something fails

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError` on a data path | `MEDTRAJECTORY_DATA_ROOT` not set or wrong; run `python -m semantic_delphi_ukb.paths` |
| `ImportError: semantic_delphi_ukb` | `src/` not on `PYTHONPATH` |
| Shell launcher starts, then dies | it expects `tmux` and a GPU-indexed queue; read the launcher |
| Metrics differ from the paper | check the split method in `prepare_manifest.json` 鈥?you may be on the legacy 80/20 profile |
| Paired comparison refuses to write | that is `verify_eval_cohorts.py` working as intended |
| Bootstrap takes hours | see the note in 搂5.2 |

---

## 8. What is deliberately not here

- **Data.** Not distributable; see [`DATA.md`](DATA.md).
- **Checkpoints and prediction dumps.** Not distributed; regenerate them.
- **`scripts/oneoff/`.** Manuscript editing, internal packaging, environment
  setup, result-registry bookkeeping, superseded ablation queues and one-off
  diagnostics. Kept for provenance, not needed for reproduction.

---

## 9. Honest limitations of this guide

1. The commands are assembled from the scripts' own CLI definitions and from the
   launchers that drove the original runs. They have **not** been run end-to-end
   here, because that requires access-controlled data and a GPU.
2. Where a script's arguments are built dynamically (for example
   `mimic_build_dataset.py`), the guide points at `--help` instead of inventing
   flags.
3. Several analyses in the manuscript were produced by multi-stage queues whose
   exact scheduling is not reconstructible. The **stages** are documented; the
   queueing is not.
