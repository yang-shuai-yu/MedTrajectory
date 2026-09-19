#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Eval design A models (vA_a0, vA_a2) under two cohort thresholds.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval_trackr.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0

run_mode () {
  local tag="$1" mh="$2" mf="$3" cap="$4"
  local out="$DATA/eval_visitA_$tag"
  mkdir -p "$out"
  echo "==== visitA mode $tag (min_history=$mh min_future=$mf cap=$cap) start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  local pids=()
  for name in vA_a0 vA_a2; do
    local ckpt="$DATA/runs/$name/checkpoints/last.pt"
    [ -f "$ckpt" ] || { echo "SKIP $name"; continue; }
    ( "$PY" "$EVAL" --ckpt "$ckpt" --data-dir "$DATA/visit_A_trackr" \
        --split test --device cuda \
        --min-history-events "$mh" --min-future-events "$mf" \
        ${cap:+--max-patients "$cap"} \
        --out-dir "$out/$name" \
        --num-rollouts 20 --max-new-tokens 30 \
        --followup-years 10 --baseline-fraction 0.65 \
        > "$out/$name.stdout.log" 2>&1 ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "==== visitA mode $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

run_mode ukb 8 3 ""
run_mode relaxed 3 2 2000
echo "ALL VISIT-A EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
