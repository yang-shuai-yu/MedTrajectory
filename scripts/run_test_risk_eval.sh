#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ROOT="${MEDTRAJECTORY_ROOT}"
cd "$ROOT"
export PYTHONPATH=src
PY=""${MEDTRAJECTORY_PYTHON:-python3}""
DATA="data/track_r_v2_1/multitype_static_prefix"
PROTO="configs/track_r_v2_2/TRACK_R_v2_2_TEST.json"
LM="results/track_r_v2_1/manifests/test_shared_landmarks.json"
OUT="results/track_r_v2_2/test"

if [ ! -f "$LM" ]; then
  "$PY" scripts/build_shared_test_landmarks.py --data-dir "$DATA" --split test --out "$LM"
fi

for s in 42 43 44; do
  for m in A0 A2 A2-noAge; do
    ckpt="results/track_r_v2_2/runs/seed${s}_additiverope1/risk/${m}/checkpoints/last.pt"
    odir="${OUT}/seed${s}/${m}"
    echo "== evaluating $m seed$s =="
    "$PY" -m semantic_delphi_ukb.evaluate_track_r \
      --family carope --model-name "$m" --checkpoint "$ckpt" \
      --protocol "$PROTO" --data-dir "$DATA" --split test \
      --landmark-manifest "$LM" --include-static-prefix true \
      --out-dir "$odir" --device cuda
  done
done

"$PY" scripts/aggregate_test_risk.py
echo "ALL_DONE"
