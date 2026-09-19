#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

root="${MEDTRAJECTORY_ROOT}"
python="${MEDTRAJECTORY_PYTHON:-python3}"
features="$root/results/track_r_v2_1/landmark_features"
output="$root/results/track_r_v2_1/runs/seed42_censorfix1/Cox-R/validation"

cd "$root"
export PYTHONPATH=src

"$python" -u -m semantic_delphi_ukb.track_r_baselines \
  --model cox \
  --train-features "$features/train.npz" \
  --val-features "$features/val.npz" \
  --eval-features "$features/val.npz" \
  --output-dir "$output"
