#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Build the Design B-matched (fixed universe + fixed split) datasets for M1 / M2 / M3.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
SPLIT=$DATA/matched_split_map.csv
cd "$DATA"

echo "=== [1/3] Bm_m1 (diagnosis) + split map ==="
"$PY" mimic_build_visit_matched.py --out-dir "$DATA/visit_Bm_m1" \
  --event-types "diagnosis" --split-map-out "$SPLIT" 2>&1 | tail -6

echo "=== [2/3] Bm_m2 (diagnosis,procedure) ==="
"$PY" mimic_build_visit_matched.py --out-dir "$DATA/visit_Bm_m2" \
  --event-types "diagnosis,procedure" --split-map-in "$SPLIT" 2>&1 | tail -6

echo "=== [3/3] Bm_m3 (diagnosis,procedure,death) ==="
"$PY" mimic_build_visit_matched.py --out-dir "$DATA/visit_Bm_m3" \
  --event-types "diagnosis,procedure,death" --split-map-in "$SPLIT" 2>&1 | tail -6

echo "=== sanity: test eid subset check ==="
for a in m1 m2; do b=$([ "$a" = "m1" ] && echo m2 || echo m3); done
"$PY" - <<'PYEOF'
import csv
from pathlib import Path
D = Path("${MEDTRAJECTORY_MIMIC_ROOT}")
sets = {}
for name in ("visit_Bm_m1", "visit_Bm_m2", "visit_Bm_m3"):
    with (D / name / "test_patient_index.csv").open(newline="") as fh:
        sets[name] = {r["eid"] for r in csv.DictReader(fh)}
print("test eids:", {k: len(v) for k, v in sets.items()})
print("m1 subset of m2:", sets["visit_Bm_m1"] <= sets["visit_Bm_m2"])
print("m2 subset of m3:", sets["visit_Bm_m2"] <= sets["visit_Bm_m3"])
print("m1∩m2∩m3:", len(sets["visit_Bm_m1"] & sets["visit_Bm_m2"] & sets["visit_Bm_m3"]))
PYEOF
echo "=== MATCHED BUILD DONE ==="
