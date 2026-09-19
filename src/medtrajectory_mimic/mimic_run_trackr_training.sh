#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Train MIMIC Track-R A0 (absolute) and A2 (additive_v2_2 relative) in parallel.
set -euo pipefail

REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
PROTO=$DATA/mimic_track_r_v2_2.json
TRACKR=$DATA/multitype_trackr
WAVE=$DATA/mimic_rope_wavelengths.json
LOGS=$DATA/logs_trackr

cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$LOGS"

launch () {
  local name="$1"; shift
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py "$@" \
    --no-tensorboard > "$LOGS/$name.log" 2>&1
}

launch trackr_a0_abs \
  --data-dir "$TRACKR" --track-r-protocol "$PROTO" \
  --include-static-prefix true --age-rope-variant legacy \
  --use-age-encoding true --use-age-rope false \
  --diseases-yaml "$YAML" \
  --run-dir "$DATA/runs/trackr_a0_abs" \
  --device cuda --max-iters 100000 --seed 42 &
P1=$!

launch trackr_a2_rel \
  --data-dir "$TRACKR" --track-r-protocol "$PROTO" \
  --include-static-prefix true --age-rope-variant additive_v2_2 \
  --use-age-encoding true --use-age-rope true \
  --rope-wavelengths-manifest "$WAVE" \
  --diseases-yaml "$YAML" \
  --run-dir "$DATA/runs/trackr_a2_rel" \
  --device cuda --max-iters 100000 --seed 42 &
P2=$!

echo "launched pids: $P1 $P2 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; echo "trackr_a0_abs done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P2; echo "trackr_a2_rel done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ALL TRACKR TRAINING DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
