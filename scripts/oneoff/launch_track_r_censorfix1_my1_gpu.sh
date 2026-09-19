#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

root="${MEDTRAJECTORY_ROOT}"
python="${MEDTRAJECTORY_PYTHON:-python3}"
data="${MEDTRAJECTORY_DATA_ROOT}"/data/track_r_v2_1/multitype_static_prefix
protocol="$root/configs/paper_protocol_v1/TRACK_R_v2_1.json"
landmarks="$root/results/track_r_v2_1/manifests/val_shared_landmarks.json"
source="$root/results/track_r_v2_1/runs/seed42_retry1"
output="$root/results/track_r_v2_1/runs/seed42_censorfix1"

cd "$root"
export PYTHONPATH=src

for model in A0-TokenStatic A1-TokenStatic A1-noStatic; do
  include_static=true
  if [[ "$model" == A1-noStatic ]]; then
    include_static=false
  fi
  "$python" -u -m semantic_delphi_ukb.evaluate_track_r \
    --family carope \
    --model-name "$model" \
    --checkpoint "$source/$model/horizon/checkpoints/best_val_horizon_auc.pt" \
    --protocol "$protocol" \
    --data-dir "$data" \
    --split val \
    --landmark-manifest "$landmarks" \
    --out-dir "$output/$model/validation" \
    --include-static-prefix "$include_static" \
    --device cuda
done
