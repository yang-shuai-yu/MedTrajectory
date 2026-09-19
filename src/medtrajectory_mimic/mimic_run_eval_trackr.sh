#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Track-R (additive_v2_2) generation eval for MIMIC A0 (absolute) and A2 (relative), in parallel.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval_trackr.py
N=${1:-1700}
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$DATA/eval_trackr"

run_one () {
  local name="$1"
  local ckpt="$DATA/runs/$name/checkpoints/last.pt"
  [ -f "$ckpt" ] || { echo "SKIP $name"; return; }
  echo "trackr-eval $name start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" "$EVAL" --ckpt "$ckpt" --data-dir "$DATA/multitype_trackr" \
    --split test --device cuda --max-patients "$N" \
    --out-dir "$DATA/eval_trackr/$name" \
    --num-rollouts 20 --max-new-tokens 30 \
    --followup-years 10 --baseline-fraction 0.65 \
    --min-history-events 8 --min-future-events 3 \
    > "$DATA/eval_trackr/$name.stdout.log" 2>&1
  echo "trackr-eval $name done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_one trackr_a0_abs &
run_one trackr_a2_rel &
wait
echo "TRACKR EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
