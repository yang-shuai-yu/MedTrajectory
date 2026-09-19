#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ROOT="${MEDTRAJECTORY_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
DATA="${MEDTRAJECTORY_EXTERNAL_ROOT}"/data/ukb_semantic_multitype_explicit_split
PGS_DATA="${MEDTRAJECTORY_DATA_ROOT}"/pgs_i21_v1
RUN_DIR=${1:?Usage: run_pgs_i21_pilot.sh RUN_DIR}

cd "$ROOT"
mkdir -p "$RUN_DIR"
"$PY" -u src/semantic_delphi_ukb/train_pgs_horizon_risk.py \
  --data-dir "$DATA" \
  --pgs-dir "$PGS_DATA" \
  --run-dir "$RUN_DIR" \
  --target-disease-ids ischemic_heart_disease,myocardial_infarction \
  --device cuda \
  --no-tensorboard \
  --epochs 30 \
  --patience 8 \
  2>&1 | tee "$RUN_DIR/stdout.log"
