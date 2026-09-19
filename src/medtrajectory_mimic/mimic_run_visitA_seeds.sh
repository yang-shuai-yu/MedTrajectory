#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Design A 3-seed completion: train A0/A2 for seeds 43 and 44 (seed 42 already done),
# as two sequential batches of two parallel jobs (2-way parallelism measured ~2.5x faster
# per model than 4-way on this single GPU).
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
WAVE=$DATA/visit_A_wavelengths.json
LOGS=$DATA/logs_visitA_seeds
mkdir -p "$LOGS"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

launch () {
  local name="$1" seed="$2" rope="$3" variant="$4" extra="$5"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/visit_A_trackr" --track-r-protocol "$DATA/visit_A_protocol.json" \
    --include-static-prefix true --age-rope-variant "$variant" \
    --use-age-encoding true --use-age-rope "$rope" \
    $extra \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed "$seed" --no-tensorboard > "$LOGS/$name.log" 2>&1
}

echo "=== batch 1: seed 43 ==="
launch vA_a0_s43 43 false legacy "" &
P1=$!
launch vA_a2_s43 43 true additive_v2_2 "--rope-wavelengths-manifest $WAVE" &
P2=$!
wait $P1; wait $P2
echo "seed 43 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== batch 2: seed 44 ==="
launch vA_a0_s44 44 false legacy "" &
P3=$!
launch vA_a2_s44 44 true additive_v2_2 "--rope-wavelengths-manifest $WAVE" &
P4=$!
wait $P3; wait $P4
echo "seed 44 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ALL DESIGN-A 3-SEED TRAINING DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
