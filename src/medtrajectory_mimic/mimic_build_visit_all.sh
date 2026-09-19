#!/bin/bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
cd "$DATA"

build () {
  local name="$1" et="$2"
  echo "=== $name ($et) ==="
  "$PY" mimic_build_visit.py --out-dir "$DATA/$name" --event-types "$et" 2>&1 | tail -8
}

build visit_B_m1 "diagnosis"
build visit_B_m2 "diagnosis,procedure"
build visit_B_m3 "diagnosis,procedure,death"
echo "ALL VISIT B BUILDS DONE"
