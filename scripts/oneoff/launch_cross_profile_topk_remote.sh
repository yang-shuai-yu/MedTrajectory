#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ROOT="${MEDTRAJECTORY_ROOT}"
PYTHON="${MEDTRAJECTORY_PYTHON:-python3}"
DATA_ROOT="${MEDTRAJECTORY_DATA_ROOT}"/paper_protocol_v1_locked_test
OUT="$ROOT/results/paper_protocol_v1/cross_profile_topk_20260810"

cd "$ROOT"
mkdir -p "$OUT"

"$PYTHON" scripts/build_shared_test_landmarks.py \
  --data-dir "$DATA_ROOT/diagnosis_death" \
  --split val \
  --seed 1337 \
  --block-size 128 \
  --out "$OUT/val_shared_landmarks_canonical.json" \
  > "$OUT/landmark_build.log" 2>&1

"$PYTHON" scripts/audit_shared_test_landmarks.py \
  --manifest "$OUT/val_shared_landmarks_canonical.json" \
  --profile "diagnosis_death=$DATA_ROOT/diagnosis_death" \
  --profile "multitype=$DATA_ROOT/multitype" \
  --out "$OUT/val_shared_landmark_audit.json" \
  --common-out "$OUT/val_shared_landmarks.json" \
  > "$OUT/landmark_audit.log" 2>&1 || test -s "$OUT/val_shared_landmarks.json"

"$PYTHON" scripts/audit_shared_test_landmarks.py \
  --manifest "$OUT/val_shared_landmarks.json" \
  --profile "diagnosis_death=$DATA_ROOT/diagnosis_death" \
  --profile "multitype=$DATA_ROOT/multitype" \
  --out "$OUT/val_shared_landmark_exact_audit.json" \
  > "$OUT/landmark_exact_audit.log" 2>&1

"$PYTHON" -u scripts/run_cross_profile_topk_comparison.py \
  --config configs/paper_protocol_v1/cross_profile_topk_v1.json \
  --output-dir "$OUT" \
  --device cuda \
  > "$OUT/stdout.log" 2>&1
