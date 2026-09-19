#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Eval visit-level B Track-R models under two cohort thresholds (UKB-aligned and relaxed).
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval_trackr.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0

MODELS=("vB_m1_a0|visit_B_m1_trackr" "vB_m2_a0|visit_B_m2_trackr" "vB_m3_a0|visit_B_m3_trackr" "vB_m3_a2|visit_B_m3_trackr")

run_mode () {
  local tag="$1" mh="$2" mf="$3" cap="$4"
  local out="$DATA/eval_visitB_$tag"
  mkdir -p "$out"
  echo "==== mode $tag (min_history=$mh min_future=$mf cap=$cap) start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  local pids=()
  for spec in "${MODELS[@]}"; do
    local name="${spec%%|*}" ddir="${spec##*|}"
    local ckpt="$DATA/runs/$name/checkpoints/last.pt"
    [ -f "$ckpt" ] || { echo "SKIP $name"; continue; }
    ( "$PY" "$EVAL" --ckpt "$ckpt" --data-dir "$DATA/$ddir" \
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
  echo "==== mode $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

run_mode ukb 8 3 ""
run_mode relaxed 3 2 2000
echo "ALL VISIT-B EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
