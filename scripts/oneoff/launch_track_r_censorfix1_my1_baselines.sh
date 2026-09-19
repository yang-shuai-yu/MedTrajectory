#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

root="${MEDTRAJECTORY_ROOT}"
ysy="${MEDTRAJECTORY_PYTHON:-python3}"
trident="${MEDTRAJECTORY_PYTHON:-python3}"
features="$root/results/track_r_v2_1/landmark_features"
output="$root/results/track_r_v2_1/runs/seed42_censorfix1"
old_logistic="$root/results/track_r_v2_1/runs/seed42/Logistic-R/validation/checkpoint.pkl"

cd "$root"
export PYTHONPATH=src

"$ysy" -u -m semantic_delphi_ukb.track_r_baselines \
  --model logistic \
  --train-features "$features/train.npz" \
  --val-features "$features/val.npz" \
  --eval-features "$features/val.npz" \
  --checkpoint "$old_logistic" \
  --predict-only \
  --output-dir "$output/Logistic-R/validation"

"$trident" -u -m semantic_delphi_ukb.track_r_baselines \
  --model cox \
  --train-features "$features/train.npz" \
  --val-features "$features/val.npz" \
  --eval-features "$features/val.npz" \
  --output-dir "$output/Cox-R/validation"
