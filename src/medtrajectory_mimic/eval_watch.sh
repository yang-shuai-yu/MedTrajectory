#!/bin/bash
# One-line status probe for the B/C evaluation chain.
cd "${MEDTRAJECTORY_MIMIC_ROOT}" || exit 1
n=$(ps -eo args | grep -c '[g]eneration_eval')
s=$(find eval_bm3_relaxed eval_B3_relaxed eval_C3_relaxed -name summary.json 2>/dev/null | wc -l)
r=$(test -f results_bc_3seeds.md && echo yes || echo no)
d=$(ps -eo args | grep -c '[m]imic_rerun_eval_bc.sh')
echo "$(date -u +%H:%M:%SZ) evals_running=$n relaxed_done=$s/21 results_md=$r driver_alive=$d"
