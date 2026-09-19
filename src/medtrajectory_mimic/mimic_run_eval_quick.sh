#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Quick-preview generation eval: all 4 models in parallel on the first N eligible test patients.
set -uo pipefail

REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval.py
N=${1:-2000}
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$DATA/eval_quick"

run_one () {
  local name="$1" ddir="$2"
  local ckpt="$DATA/runs/$name/checkpoints/last.pt"
  [ -f "$ckpt" ] || { echo "SKIP $name"; return; }
  echo "quick $name start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" "$EVAL" \
    --ckpt "$ckpt" --data-dir "$DATA/$ddir" \
    --split test --device cuda --max-patients "$N" \
    --out-dir "$DATA/eval_quick/$name" \
    --num-rollouts 20 --max-new-tokens 30 \
    --followup-years 10 --baseline-fraction 0.65 \
    --min-history-events 8 --min-future-events 3 \
    > "$DATA/eval_quick/$name.stdout.log" 2>&1
  echo "quick $name done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_one m1_abs multitype_m1 &
run_one m2_abs multitype_m2 &
run_one m3_abs multitype &
run_one m3_rel multitype &
wait
echo "QUICK EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
