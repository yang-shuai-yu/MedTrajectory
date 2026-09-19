#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# MIMIC-IV CARoPE training driver: smoke-test relative config, then 4 full runs.
# Experiment 1 (multitype ablation, absolute time): M1 / M2 / M3
# Experiment 2 (temporal, on M3): Absolute (m3_abs) vs Relative (m3_rel, legacy RoPE)
set -euo pipefail

REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs

cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$LOGS"

# ---- Smoke test: relative (legacy) config on the simple multitype format ----
echo "==================== smoke_rel (relative legacy) ===================="
"$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
  --data-dir "$DATA/multitype" \
  --diseases-yaml "$YAML" \
  --run-dir "$DATA/runs/smoke_rel" \
  --device cuda --max-iters 5 --seed 42 \
  --use-age-encoding true --use-age-rope true --age-rope-variant legacy \
  --no-tensorboard 2>&1 | tee "$LOGS/smoke_rel.log"

# ---- Full training ----
for spec in \
  "m1_abs|$DATA/multitype_m1|false" \
  "m2_abs|$DATA/multitype_m2|false" \
  "m3_abs|$DATA/multitype|false" \
  "m3_rel|$DATA/multitype|true" ; do
  IFS='|' read -r name ddir rope <<< "$spec"
  echo "==================== $name start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===================="
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$ddir" \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed 42 \
    --use-age-encoding true --use-age-rope "$rope" \
    --age-rope-variant legacy \
    --no-tensorboard 2>&1 | tee "$LOGS/$name.log"
  echo "==================== $name done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===================="
done

echo "ALL TRAINING DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
