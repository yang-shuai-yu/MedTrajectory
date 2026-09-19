#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Auto-transition: wait for all 4 CARoPE training runs to finish, then run eval.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"

echo "watcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f train_car_rope_pretraining > /dev/null 2>&1; do
  sleep 300
done
echo "all training finished at $(date -u +%Y-%m-%dT%H:%M:%SZ), launching eval..."

bash "$DATA/mimic_run_eval.sh" > "$DATA/eval_master.log" 2>&1
echo "eval finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# compile results table
cd "$DATA"
"${MEDTRAJECTORY_PYTHON:-python3}" "$DATA/mimic_compile_results.py" > "$DATA/results_table.md" 2>&1
echo "compiled results table at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# statistical analysis (mean +/- SD, Exp2 paired Wilcoxon)
"${MEDTRAJECTORY_PYTHON:-python3}" "$DATA/mimic_analyze_results.py" > "$DATA/results_analysis.md" 2>&1
echo "analysis done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
