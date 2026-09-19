#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ROOT="${MEDTRAJECTORY_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
RUN_DIR=${1:?Usage: run_pgs_multiancestry_i21.sh RUN_DIR}

cd "$ROOT"
mkdir -p "$RUN_DIR"
"$PY" -u src/semantic_delphi_ukb/train_pgs_multiancestry_horizon_risk.py \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --no-tensorboard \
  --train-batch-size 512 \
  --epochs 40 \
  --patience 10 \
  2>&1 | tee "$RUN_DIR/stdout.log"
