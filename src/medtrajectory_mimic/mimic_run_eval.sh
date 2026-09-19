#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Run generation evaluation on all 4 trained CARoPE models (test split).
set -euo pipefail

REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0

for spec in \
  "m1_abs|multitype_m1" \
  "m2_abs|multitype_m2" \
  "m3_abs|multitype" \
  "m3_rel|multitype" ; do
  IFS='|' read -r name ddir <<< "$spec"
  CKPT="$DATA/runs/$name/checkpoints/last.pt"
  if [ ! -f "$CKPT" ]; then
    echo "==================== eval $name SKIPPED (no checkpoint) ===================="
    continue
  fi
  echo "==================== eval $name start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===================="
  "$PY" "$EVAL" \
    --ckpt "$CKPT" \
    --data-dir "$DATA/$ddir" \
    --split test --device cuda \
    --out-dir "$DATA/eval/$name" \
    --num-rollouts 20 --max-new-tokens 30 \
    --followup-years 10 --baseline-fraction 0.65 \
    --min-history-events 8 --min-future-events 3 \
    2>&1 | tee "$DATA/eval/$name.stdout.log" || echo "eval $name FAILED"
  echo "==================== eval $name done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===================="
done

echo "ALL EVAL DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
