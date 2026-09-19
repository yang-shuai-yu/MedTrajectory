#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Build Track-R static-prefix overlays for the visit-level (design B) datasets.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
PAT=$DATA/mimic-iv-3.1/hosp/patients.csv.gz

for spec in "visit_B_m1" "visit_B_m2" "visit_B_m3"; do
  echo "=== trackr overlay $spec ==="
  "$PY" "$DATA/mimic_build_track_r.py" \
    --source-data-dir "$DATA/$spec" \
    --patients-csv "$PAT" \
    --output-dir "$DATA/${spec}_trackr" 2>&1 | tail -4
done

echo "=== B-M3 wavelengths ==="
cd "${MEDTRAJECTORY_ROOT}"
"$PY" scripts/build_track_r_v2_2_wavelengths.py \
  --protocol configs/track_r_v2_2/TRACK_R_v2_2.json \
  --data-dir "$DATA/visit_B_m3_trackr" \
  --output "$DATA/visit_B_m3_wavelengths.json" 2>&1 | tail -3
echo "ALL B TRACKR BUILDS DONE"
