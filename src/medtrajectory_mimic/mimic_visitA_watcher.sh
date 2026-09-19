#!/bin/bash
# Wait for design A training, then eval both thresholds.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
echo "visitA watcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f train_car_rope_pretraining > /dev/null 2>&1; do
  sleep 180
done
echo "visit-A training finished $(date -u +%Y-%m-%dT%H:%M:%SZ) -> evals"
bash "$DATA/mimic_run_eval_visitA.sh" > "$DATA/eval_visitA_master.log" 2>&1
echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
