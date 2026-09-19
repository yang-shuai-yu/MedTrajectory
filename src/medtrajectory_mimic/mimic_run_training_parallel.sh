#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Parallel CARoPE training driver: all 4 models at once (tiny models, ~3.6GB each, 24GB GPU).
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

launch () {
  local name="$1" ddir="$2" rope="$3"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$ddir" \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed 42 \
    --use-age-encoding true --use-age-rope "$rope" \
    --age-rope-variant legacy \
    --no-tensorboard > "$LOGS/$name.log" 2>&1
}

launch m1_abs "$DATA/multitype_m1" false &
P1=$!
launch m2_abs "$DATA/multitype_m2" false &
P2=$!
launch m3_abs "$DATA/multitype" false &
P3=$!
launch m3_rel "$DATA/multitype" true &
P4=$!

echo "launched pids: $P1 $P2 $P3 $P4 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; echo "m1_abs done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P2; echo "m2_abs done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P3; echo "m3_abs done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P4; echo "m3_rel done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ALL TRAINING DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
