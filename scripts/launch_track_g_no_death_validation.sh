#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ROOT="${MEDTRAJECTORY_ROOT}"
PYTHON="${MEDTRAJECTORY_PYTHON:-python3}"
OUTPUT_ROOT="$ROOT/results/track_g_v1/val_no_death_token"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$PYTHON" scripts/run_track_g_no_death_validation.py --execute \
  >>"$OUTPUT_ROOT/queue.log" 2>&1
