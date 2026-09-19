#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Patch Track-R manifests for visit-level B datasets and launch 4 training runs in parallel.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
PROTO=$DATA/visit_B_m3_protocol.json
WAVE=$DATA/visit_B_m3_wavelengths.json
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_visitB
mkdir -p "$LOGS"

# --- patch manifests ---
cat > /tmp/patch_manifest.py <<'PYEOF'
import json, sys
ROOT = "${MEDTRAJECTORY_ROOT}"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol
proto = load_track_r_protocol("${MEDTRAJECTORY_MIMIC_ROOT}"/visit_B_m3_protocol.json)
for name in ("visit_B_m1_trackr", "visit_B_m2_trackr", "visit_B_m3_trackr"):
    p = f"${MEDTRAJECTORY_MIMIC_ROOT}"/{name}/prepare_manifest.json
    m = json.load(open(p, encoding="utf-8"))
    m["protocol_id"] = proto["data_protocol_id"]
    m["loss_contract"] = proto["loss_contract"]
    m["static_prefix"] = proto["static_prefix"]
    m["dynamic_bos"] = proto["dynamic_bos"]
    json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
    print("patched", name)
PYEOF
"$PY" /tmp/patch_manifest.py

cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

launch () {
  local name="$1" ddir="$2" rope="$3" variant="$4" extra="$5"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/$ddir" --track-r-protocol "$PROTO" \
    --include-static-prefix true --age-rope-variant "$variant" \
    --use-age-encoding true --use-age-rope "$rope" \
    $extra \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed 42 \
    --no-tensorboard > "$LOGS/$name.log" 2>&1
}

launch vB_m1_a0 visit_B_m1_trackr false legacy "" &
P1=$!
launch vB_m2_a0 visit_B_m2_trackr false legacy "" &
P2=$!
launch vB_m3_a0 visit_B_m3_trackr false legacy "" &
P3=$!
launch vB_m3_a2 visit_B_m3_trackr true additive_v2_2 "--rope-wavelengths-manifest $WAVE" &
P4=$!

echo "launched pids: $P1 $P2 $P3 $P4 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; echo "vB_m1_a0 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P2; echo "vB_m2_a0 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P3; echo "vB_m3_a0 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P4; echo "vB_m3_a2 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ALL VISIT-B TRAINING DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
