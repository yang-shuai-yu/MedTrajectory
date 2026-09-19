#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Design A build only (A1-A5): data, track-R overlay, wavelengths, protocol, manifest patch.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
PAT=$DATA/mimic-iv-3.1/hosp/patients.csv.gz

echo "=== [A1] build visit_A (principal diagnosis + death) ==="
"$PY" "$DATA/mimic_build_visit.py" --out-dir "$DATA/visit_A" --event-types "diagnosis,death" 2>&1 | tail -6

echo "=== [A2] track-R static overlay ==="
"$PY" "$DATA/mimic_build_track_r.py" --source-data-dir "$DATA/visit_A" \
  --patients-csv "$PAT" --output-dir "$DATA/visit_A_trackr" 2>&1 | tail -3

echo "=== [A3] wavelengths ==="
cd "$REPO"
"$PY" scripts/build_track_r_v2_2_wavelengths.py --protocol configs/track_r_v2_2/TRACK_R_v2_2.json \
  --data-dir "$DATA/visit_A_trackr" --output "$DATA/visit_A_wavelengths.json" 2>&1 | tail -3

echo "=== [A4] protocol ==="
"$PY" "$DATA/mimic_make_protocol_generic.py" --wavelength "$DATA/visit_A_wavelengths.json" \
  --out "$DATA/visit_A_protocol.json" --output-root "$DATA/results/visit_A" 2>&1 | tail -2

echo "=== [A5] patch manifest ==="
cat > /tmp/patch_va.py <<'PYEOF'
import json, sys
ROOT = "${MEDTRAJECTORY_ROOT}"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol
proto = load_track_r_protocol("${MEDTRAJECTORY_MIMIC_ROOT}"/visit_A_protocol.json)
p = "${MEDTRAJECTORY_MIMIC_ROOT}"/visit_A_trackr/prepare_manifest.json
m = json.load(open(p, encoding="utf-8"))
m["protocol_id"] = proto["data_protocol_id"]
m["loss_contract"] = proto["loss_contract"]
m["static_prefix"] = proto["static_prefix"]
m["dynamic_bos"] = proto["dynamic_bos"]
json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
print("patched visit_A_trackr")
PYEOF
"$PY" /tmp/patch_va.py
echo "=== VISIT-A BUILD DONE ==="
