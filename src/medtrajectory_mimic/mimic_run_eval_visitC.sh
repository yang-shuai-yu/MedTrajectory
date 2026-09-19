#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Eval design C models (vC_a0, vC_a2) under two cohort thresholds, then analyze.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval_trackr.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0

run_mode () {
  local tag="$1" mh="$2" mf="$3" cap="$4"
  local out="$DATA/eval_visitC_$tag"
  mkdir -p "$out"
  echo "==== visitC mode $tag (min_history=$mh min_future=$mf cap=$cap) start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  local pids=()
  for name in vC_a0 vC_a2; do
    local ckpt="$DATA/runs/$name/checkpoints/last.pt"
    if [ ! -f "$ckpt" ]; then echo "SKIP $name (no checkpoint)"; continue; fi
    ( "$PY" "$EVAL" --ckpt "$ckpt" --data-dir "$DATA/visit_C_trackr" \
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
  echo "==== visitC mode $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

run_mode ukb 8 3 ""
run_mode relaxed 3 2 2000
echo "ALL VISIT-C EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== running design C analysis $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" "$DATA/mimic_analyze_visitC.py" > "$DATA/results_visitC.md" 2>&1 || echo "ANALYSIS FAILED"
echo "=== design C results written to $DATA/results_visitC.md $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
