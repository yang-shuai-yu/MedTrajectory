# MIMIC-IV pipeline (to be collected)

This directory is a placeholder. The MIMIC-IV cross-setting replication pipeline
is **not** in the UKB repository; it lives on the compute host at:

```text
MIMIC_ROOT
```

## What belongs here

Roughly 40 top-level `.py` / `.sh` files, including:

| Stage | Scripts |
|---|---|
| Dataset build | `mimic_build_dataset.py` (event level), `mimic_build_visit.py` (visit level, design B), `mimic_build_track_r.py` (Track-R static prefix + anchor) |
| Semantic initialisation | `mimic_qwen_embed.py` (Qwen text-embedding-v4 -> PCA 64), `mimic_semantic_*` helpers |
| Training drivers | `mimic_run_training.sh`, `mimic_run_training_parallel.sh`, `mimic_run_trackr_training.sh`, `mimic_train_matched_baseline.py` |
| Evaluation | `mimic_generation_eval.py`, `mimic_generation_eval_trackr.py`, `mimic_run_eval*.sh` |
| Analysis / pairing | `mimic_paired_3seed.py`, `mimic_ethos_paired.py`, `mimic_final_conclusions.py`, `verify_eval_cohorts.py` |

## Collection procedure (key never leaves the server)

Run `collect_mimic.sh`: it rewrites every copy with `sed`, replacing the hardcoded
DashScope key with `os.environ.get("DASHSCOPE_API_KEY", "")` **on the server**, then
tars the result. The script aborts if the key pattern survives in the staged copies.

**Before that, rotate the key** — it existed in plaintext on the server, so it must
be treated as compromised regardless of how the files are handled.

## Data note

MIMIC-IV and UK Biobank data are **not** distributable with this code. Both require
their own credentialed access. No row-level data, checkpoints, or prediction dumps
are included in this repository.
