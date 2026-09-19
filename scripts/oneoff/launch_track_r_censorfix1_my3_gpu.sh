#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

root="${MEDTRAJECTORY_ROOT}"
python="${MEDTRAJECTORY_PYTHON:-python3}"
data="${MEDTRAJECTORY_DATA_ROOT}"/data/track_r_v2_1/multitype_static_prefix
protocol="$root/configs/paper_protocol_v1/TRACK_R_v2_1.json"
landmarks="$root/results/track_r_v2_1/manifests/val_shared_landmarks.json"
features="$root/results/track_r_v2_1/landmark_features"
output="$root/results/track_r_v2_1/runs/seed42_censorfix1"
diagnostics="$root/results/track_r_v2_1/diagnostics/carope_static_integration/seed42_diagnostic2"
source="$root/results/track_r_v2_1/runs/seed42_retry1"

cd "$root"
export PYTHONPATH=src

"$python" -u -m semantic_delphi_ukb.track_r_baselines \
  --model mdrmf-clinical \
  --train-features "$features/train.npz" \
  --val-features "$features/val.npz" \
  --eval-features "$features/val.npz" \
  --output-dir "$output/MDRMF-Clinical-R/validation" \
  --device cuda

"$python" -u -m semantic_delphi_ukb.evaluate_track_r \
  --family carope \
  --model-name A0-noStatic \
  --checkpoint "$diagnostics/A0-noStatic/horizon/checkpoints/best_val_horizon_auc.pt" \
  --protocol "$protocol" \
  --data-dir "$data" \
  --split val \
  --landmark-manifest "$landmarks" \
  --out-dir "$output/A0-noStatic/validation" \
  --include-static-prefix false \
  --device cuda

for model in Med-BERT-Paper Med-BERT-Matched-S; do
  "$python" -u -m semantic_delphi_ukb.evaluate_track_r \
    --family medbert \
    --model-name "$model" \
    --checkpoint "$source/$model/horizon/checkpoints/best_val_horizon_auc.pt" \
    --protocol "$protocol" \
    --data-dir "$data" \
    --split val \
    --landmark-manifest "$landmarks" \
    --out-dir "$output/$model/validation" \
    --include-static-prefix true \
    --device cuda
done
