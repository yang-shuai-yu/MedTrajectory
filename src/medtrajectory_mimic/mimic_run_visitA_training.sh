#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Design A training only: vA_a0 (absolute) + vA_a2 (additive_v2_2) in parallel.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_visitA
mkdir -p "$LOGS"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

launch () {
  local name="$1" rope="$2" variant="$3" extra="$4"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/visit_A_trackr" --track-r-protocol "$DATA/visit_A_protocol.json" \
    --include-static-prefix true --age-rope-variant "$variant" \
    --use-age-encoding true --use-age-rope "$rope" \
    $extra \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed 42 --no-tensorboard > "$LOGS/$name.log" 2>&1
}

launch vA_a0 false legacy "" &
P1=$!
launch vA_a2 true additive_v2_2 "--rope-wavelengths-manifest $DATA/visit_A_wavelengths.json" &
P2=$!
echo "launched visit-A pids: $P1 $P2 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; echo "vA_a0 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P2; echo "vA_a2 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ALL VISIT-A TRAINING DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
