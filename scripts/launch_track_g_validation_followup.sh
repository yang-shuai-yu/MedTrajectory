#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ROOT="${MEDTRAJECTORY_ROOT}"
PYTHON="${MEDTRAJECTORY_PYTHON:-python3}"

cd "$ROOT"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$PYTHON" scripts/orchestrate_track_g_validation_followup.py \
  >>results/track_g_v1/validation_followup.log 2>&1
