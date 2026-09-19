#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Wait for the Design A 3-seed training to finish, then eval all seeds and summarize.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
echo "visitA-seeds watcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f train_car_rope_pretraining > /dev/null 2>&1; do
  sleep 180
done
echo "all training finished $(date -u +%Y-%m-%dT%H:%M:%SZ) -> running seed evals"
bash "$DATA/mimic_run_eval_visitA_seeds.sh" > "$DATA/eval_visitA_seeds_master.log" 2>&1
echo "evals done $(date -u +%Y-%m-%dT%H:%M:%SZ) -> summarizing"
"$PY" "$DATA/mimic_analyze_visitA_seeds.py" > "$DATA/results_visitA_3seeds.md" 2>&1 || echo "ANALYSIS FAILED"
echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
