#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Provenance certificate: does the torch intra-op thread count change the training
# trajectory?  Same data / protocol / seed / iteration budget, three runs:
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

run () {
  local tag="$1" threads="$2"
  local rd=/tmp/bitcheck/$tag
  rm -rf "$rd"
  OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/visit_B_m3_trackr" --track-r-protocol "$DATA/visit_B_m3_protocol.json" \
    --include-static-prefix true --age-rope-variant legacy \
    --use-age-encoding true --use-age-rope false \
    --diseases-yaml "$DATA/multitype/mimic_diseases.yaml" \
    --run-dir "$rd" --device cuda --max-iters "$ITERS" --eval-interval 100000 \
    --seed 44 --no-tensorboard > "/tmp/bitcheck_$tag.log" 2>&1
  echo "$tag exit=$?"
}

mkdir -p /tmp/bitcheck
echo "=== certificate start $(date -u +%Y-%m-%dT%H:%M:%SZ) iters=$ITERS ==="
/usr/bin/time -f "t1_A wall=%es" bash -c 'true' 2>/dev/null
run t1_A 1
run t1_B 1
run t104_A 104
echo "=== weight hashes ==="
for tag in t1_A t1_B t104_A; do
  "$PY" "$DATA/hash_ckpt.py" "/tmp/bitcheck/$tag/checkpoints/last.pt"
done
echo "=== certificate done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
