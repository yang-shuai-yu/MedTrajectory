#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Run generation evaluation for several models IN PARALLEL (each is single-core CPU bound).
set -uo pipefail

REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$DATA/eval"

run_one () {
  local name="$1" ddir="$2"
  local ckpt="$DATA/runs/$name/checkpoints/last.pt"
  if [ ! -f "$ckpt" ]; then echo "SKIP $name (no ckpt)"; return; fi
  echo "eval $name start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" "$EVAL" \
    --ckpt "$ckpt" --data-dir "$DATA/$ddir" \
    --split test --device cuda --out-dir "$DATA/eval/$name" \
    --num-rollouts 20 --max-new-tokens 30 \
    --followup-years 10 --baseline-fraction 0.65 \
    --min-history-events 8 --min-future-events 3 \
    > "$DATA/eval/$name.stdout.log" 2>&1
  echo "eval $name done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_one m2_abs multitype_m2 &
P1=$!
run_one m3_abs multitype &
P2=$!
run_one m3_rel multitype &
P3=$!

echo "parallel eval launched: $P1 $P2 $P3 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; wait $P2; wait $P3
echo "ALL PARALLEL EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
