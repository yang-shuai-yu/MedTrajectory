#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Eval Design A for all three seeds (42/43/44) x {A0, A2} under both cohort thresholds,
# in batches of 3 to limit single-GPU contention.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval_trackr.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0

# seed|a0_name|a2_name
SEEDS=("42|vA_a0|vA_a2" "43|vA_a0_s43|vA_a2_s43" "44|vA_a0_s44|vA_a2_s44")

run_mode () {
  local tag="$1" mh="$2" mf="$3" cap="$4"
  local out="$DATA/eval_visitA_seeds_$tag"
  mkdir -p "$out"
  echo "==== seeds mode $tag (mh=$mh mf=$mf cap=$cap) start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  local pids=()
  for spec in "${SEEDS[@]}"; do
    IFS='|' read -r seed a0 a2 <<< "$spec"
    for pair in "$a0" "$a2"; do
      local ckpt="$DATA/runs/$pair/checkpoints/last.pt"
      if [ ! -f "$ckpt" ]; then echo "SKIP $pair (no ckpt)"; continue; fi
      ( "$PY" "$EVAL" --ckpt "$ckpt" --data-dir "$DATA/visit_A_trackr" \
          --split test --device cuda \
          --min-history-events "$mh" --min-future-events "$mf" \
          ${cap:+--max-patients "$cap"} \
          --out-dir "$out/$pair" \
          --num-rollouts 20 --max-new-tokens 30 \
          --followup-years 10 --baseline-fraction 0.65 \
          > "$out/$pair.stdout.log" 2>&1 ) &
      pids+=($!)
      # batch of 3
      if [ "${#pids[@]}" -ge 3 ]; then
        for p in "${pids[@]}"; do wait "$p"; done
        pids=()
      fi
    done
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "==== seeds mode $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

run_mode ukb 8 3 ""
run_mode relaxed 3 2 2000
echo "ALL SEED EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
