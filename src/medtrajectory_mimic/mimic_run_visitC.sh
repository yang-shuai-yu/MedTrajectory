#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Design C: track-R overlay + wavelengths + protocol + manifest patch, then train vC_a0 / vC_a2.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
PAT=$DATA/mimic-iv-3.1/hosp/patients.csv.gz
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_visitC
mkdir -p "$LOGS"

echo "=== [C1] track-R overlay ==="
"$PY" "$DATA/mimic_build_track_r.py" --source-data-dir "$DATA/visit_C" \
  --patients-csv "$PAT" --output-dir "$DATA/visit_C_trackr" 2>&1 | tail -3

echo "=== [C2] wavelengths ==="
cd "$REPO"
"$PY" scripts/build_track_r_v2_2_wavelengths.py --protocol configs/track_r_v2_2/TRACK_R_v2_2.json \
  --data-dir "$DATA/visit_C_trackr" --output "$DATA/visit_C_wavelengths.json" 2>&1 | tail -3

echo "=== [C3] protocol ==="
"$PY" "$DATA/mimic_make_protocol_generic.py" --wavelength "$DATA/visit_C_wavelengths.json" \
  --out "$DATA/visit_C_protocol.json" --output-root "$DATA/results/visit_C" 2>&1 | tail -2

echo "=== [C4] patch manifest ==="
cat > /tmp/patch_vc.py <<'PYEOF'
import json, sys
ROOT = "${MEDTRAJECTORY_ROOT}"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol
proto = load_track_r_protocol("${MEDTRAJECTORY_MIMIC_ROOT}"/visit_C_protocol.json)
p = "${MEDTRAJECTORY_MIMIC_ROOT}"/visit_C_trackr/prepare_manifest.json
m = json.load(open(p, encoding="utf-8"))
m["protocol_id"] = proto["data_protocol_id"]
m["loss_contract"] = proto["loss_contract"]
m["static_prefix"] = proto["static_prefix"]
m["dynamic_bos"] = proto["dynamic_bos"]
json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
print("patched visit_C_trackr")
PYEOF
"$PY" /tmp/patch_vc.py

echo "=== [C5] train vC_a0 / vC_a2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

launch () {
  local name="$1" rope="$2" variant="$3" extra="$4"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/visit_C_trackr" --track-r-protocol "$DATA/visit_C_protocol.json" \
    --include-static-prefix true --age-rope-variant "$variant" \
    --use-age-encoding true --use-age-rope "$rope" \
    $extra \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed 42 --no-tensorboard > "$LOGS/$name.log" 2>&1
}

launch vC_a0 false legacy "" &
P1=$!
launch vC_a2 true additive_v2_2 "--rope-wavelengths-manifest $DATA/visit_C_wavelengths.json" &
P2=$!
echo "launched visit-C pids: $P1 $P2 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; echo "vC_a0 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P2; echo "vC_a2 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ALL VISIT-C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
