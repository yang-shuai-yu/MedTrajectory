#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Provenance certificate, second attempt.
#
# The first attempt used --eval-interval 100000, which makes the training loop
# save checkpoints only at iteration 0 (0 % 100000 == 0), so all three hashes
# were the untrained initialisation.  Here --eval-interval 500 forces a save at
# iteration 3000 and also records the val loss at that iteration.
#
#   t1_A, t1_B  (repeat at 1 thread)   -> isolates GPU-level nondeterminism
#   t104_A      (default 104 threads)  -> compared against the t1 pair
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0
ITERS=${ITERS:-3000}
EVAL_EVERY=${EVAL_EVERY:-500}

run () {
  local tag="$1" threads="$2"
  local rd=/tmp/bitcheck2/$tag
  rm -rf "$rd"
  local t0 t1
  t0=$(date +%s)
  OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/visit_B_m3_trackr" --track-r-protocol "$DATA/visit_B_m3_protocol.json" \
    --include-static-prefix true --age-rope-variant legacy \
    --use-age-encoding true --use-age-rope false \
    --diseases-yaml "$DATA/multitype/mimic_diseases.yaml" \
    --run-dir "$rd" --device cuda --max-iters "$ITERS" --eval-interval "$EVAL_EVERY" \
    --seed 44 --no-tensorboard > "/tmp/bitcheck2_$tag.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  local wall=$((t1 - t0))
  printf "%-8s threads=%-4s exit=%s wall=%ss  step_ms=%s  val=%s\n" \
    "$tag" "$threads" "$rc" "$wall" \
    "$("$PY" -c "import json,sys;d=json.load(open('$rd/status.json'));print(round(d.get('performance_step_time_seconds',-1)*1000,2))")" \
    "$("$PY" -c "import json,sys;d=json.load(open('$rd/status.json'));print(d.get('best_val_pretraining_loss'))")"
}

mkdir -p /tmp/bitcheck2
echo "=== certificate start $(date -u +%Y-%m-%dT%H:%M:%SZ) iters=$ITERS eval_every=$EVAL_EVERY ==="
run t1_A 1
run t1_B 1
run t104_A 104
echo "=== weight hashes at iteration $ITERS ==="
for tag in t1_A t1_B t104_A; do
  "$PY" "$DATA/hash_ckpt.py" "/tmp/bitcheck2/$tag/checkpoints/last.pt"
done
echo "=== certificate done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
