#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Measure AGGREGATE evaluation throughput versus concurrency.
#
# Motivation: a single 40-patient probe took 0.425 s/patient on an idle machine
# but 4.825 s/patient with 14 other evals running -- an 11x per-process
# slowdown.  Small probes are dominated by ~8 s of fixed startup, so this
# benchmark uses N=200 patients per process and sweeps K = 1, 3, 6, 12
# concurrent processes, reporting aggregate throughput so the optimal
# concurrency can be chosen from measurement rather than assumption.
set -uo pipefail
cd "${MEDTRAJECTORY_MIMIC_ROOT}"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${MEDTRAJECTORY_ROOT}":"${MEDTRAJECTORY_ROOT}"/src
PY="${MEDTRAJECTORY_PYTHON:-python3}"
N=${N:-200}
BENCH=/tmp/concbench

run_one () {
  local idx=$1
  "$PY" mimic_generation_eval_trackr.py \
    --ckpt runs/bm_m1_a0/checkpoints/last.pt \
    --data-dir visit_Bm_m1_trackr --split test --device cuda \
    --min-history-events 3 --min-future-events 2 --max-patients "$N" \
    --eid-file matched_bm_relaxed.txt --out-dir "$BENCH/$idx" \
    --num-rollouts 20 --max-new-tokens 30 --followup-years 10 \
    --baseline-fraction 0.65 > "$BENCH/$idx.log" 2>&1
}

echo "N=$N patients per process"
echo "K  wall_s  per_process_s_per_patient  aggregate_patient_per_s"
for K in 1 3 6 12; do
  rm -rf "$BENCH"
  mkdir -p "$BENCH"
  S=$(date +%s)
  for i in $(seq 1 "$K"); do run_one "$i" & done
  wait
  E=$(date +%s)
  W=$((E - S))
  "$PY" - "$K" "$N" "$W" <<'PY'
import sys
k, n, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print(f"{k:<3d}{w:<8d}{w / n:<28.3f}{k * n / w:.2f}")
PY
done
