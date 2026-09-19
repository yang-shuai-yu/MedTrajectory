#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Measure the true per-patient cost of one generation evaluation under the
# current 14-way load, using a 40-patient probe.
set -uo pipefail
cd "${MEDTRAJECTORY_MIMIC_ROOT}"
rm -rf /tmp/probe_eval
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${MEDTRAJECTORY_ROOT}":"${MEDTRAJECTORY_ROOT}"/src
N=${1:-40}
S=$(date +%s)
"${MEDTRAJECTORY_PYTHON:-python3}" mimic_generation_eval_trackr.py \
  --ckpt runs/bm_m1_a0/checkpoints/last.pt \
  --data-dir visit_Bm_m1_trackr --split test --device cuda \
  --min-history-events 3 --min-future-events 2 --max-patients "$N" \
  --eid-file matched_bm_relaxed.txt --out-dir /tmp/probe_eval \
  --num-rollouts 20 --max-new-tokens 30 --followup-years 10 \
  --baseline-fraction 0.65 > /tmp/probe_eval.log 2>&1
rc=$?
E=$(date +%s)
WALL=$((E - S))
CASES=$("${MEDTRAJECTORY_PYTHON:-python3}" - <<'PY'
import json
try:
    d = json.load(open("/tmp/probe_eval/summary.json"))
    print(d.get("num_cases"))
except Exception as exc:
    print("ERR:" + str(exc))
PY
)
echo "probe exit=$rc wall=${WALL}s cases=$CASES"
if [ "$CASES" != "ERR" ] && [ -n "$CASES" ] && [ "$CASES" != "None" ]; then
  echo "per-patient = $("${MEDTRAJECTORY_PYTHON:-python3}" -c "print(round($WALL/$CASES,3))") s"
  echo "2000-patient job = $("${MEDTRAJECTORY_PYTHON:-python3}" -c "print(round(2000*$WALL/$CASES/60,1))") min"
  echo "aggregate at 14-way = $("${MEDTRAJECTORY_PYTHON:-python3}" -c "print(round(14*$CASES/$WALL,2))") patient-evals/s"
fi
