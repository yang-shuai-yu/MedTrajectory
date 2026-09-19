#!/bin/bash
# Queue: wait for the ETHOS-Matched pipeline, then run the 3-seed completion for
# Bm-matched multitype (Exp1) and Design B / C Experiment 2, then evaluate + summarise.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
echo "bc-seeds launcher start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
waited=0
while pgrep -f mimic_run_ethos > /dev/null 2>&1 \
      || pgrep -f train_car_rope_pretraining > /dev/null 2>&1 \
      || pgrep -f mimic_train_matched_baseline > /dev/null 2>&1 \
      || pgrep -f generation_eval > /dev/null 2>&1; do
  sleep 180
  waited=$((waited + 180))
  if [ "$waited" -gt 259200 ]; then echo "WARN: timeout (72h)"; break; fi
done
echo "=== GPU free, starting BC 3-seed training $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash "$DATA/mimic_run_bc_seeds.sh" > "$DATA/bc_seeds_master.log" 2>&1
echo "=== BC training done, starting evals $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash "$DATA/mimic_run_eval_bc_seeds.sh" > "$DATA/eval_bc_seeds_master.log" 2>&1
echo "=== ALL BC 3-SEED WORK DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
