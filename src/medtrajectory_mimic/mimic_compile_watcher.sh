#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Wait for the fixed eval re-run to finish, then compile + analyze results.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"

echo "compile-watcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f mimic_generation_eval > /dev/null 2>&1; do
  sleep 120
done
echo "eval finished at $(date -u +%Y-%m-%dT%H:%M:%SZ), compiling..."

cd "$DATA"
"${MEDTRAJECTORY_PYTHON:-python3}" "$DATA/mimic_compile_results.py" > "$DATA/results_table.md" 2>&1
"${MEDTRAJECTORY_PYTHON:-python3}" "$DATA/mimic_analyze_results.py" > "$DATA/results_analysis.md" 2>&1
echo "compile+analyze done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
