#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Wait for visit-B training to finish, then run both-threshold evals + analysis.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"

echo "visitB eval watcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f train_car_rope_pretraining > /dev/null 2>&1; do
  sleep 180
done
echo "visit-B training finished $(date -u +%Y-%m-%dT%H:%M:%SZ) -> running evals"
bash "$DATA/mimic_run_eval_visitB.sh" > "$DATA/eval_visitB_master.log" 2>&1

PY="${MEDTRAJECTORY_PYTHON:-python3}"
"$PY" "$DATA/mimic_analyze_visitB.py" > "$DATA/results_visitB.md" 2>&1 || echo "analysis failed"
echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
