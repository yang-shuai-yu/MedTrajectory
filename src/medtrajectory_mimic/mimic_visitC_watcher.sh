#!/bin/bash
# Wait for design C training, then eval both thresholds.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
echo "visitC watcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f train_car_rope_pretraining > /dev/null 2>&1; do
  sleep 180
done
echo "visit-C training finished $(date -u +%Y-%m-%dT%H:%M:%SZ) -> evals"
bash "$DATA/mimic_run_eval_visitC.sh" > "$DATA/eval_visitC_master.log" 2>&1
echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
