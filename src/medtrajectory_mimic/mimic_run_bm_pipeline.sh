#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Design B-matched pipeline: track-R overlays + wavelengths + protocol + manifest patches (CPU, now),
# then WAIT for the Design A 3-seed driver to finish, then train Bm M1/M2/M3 and run the
# matched-patient multitype comparison.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
PAT=$DATA/mimic-iv-3.1/hosp/patients.csv.gz
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_bm
mkdir -p "$LOGS"

# ---------------- CPU phase: track-R overlays ----------------
for spec in "visit_Bm_m1:1806" "visit_Bm_m2:4172" "visit_Bm_m3:4172"; do
  name="${spec%%:*}"
  if [ -d "$DATA/${name}_trackr" ]; then echo "skip overlay $name (exists)"; continue; fi
  echo "=== overlay $name ==="
  "$PY" "$DATA/mimic_build_track_r.py" --source-data-dir "$DATA/$name" \
    --patients-csv "$PAT" --output-dir "$DATA/${name}_trackr" 2>&1 | tail -2
done

echo "=== wavelengths (Bm_m3) ==="
cd "$REPO"
if [ ! -f "$DATA/visit_Bm_m3_wavelengths.json" ]; then
  "$PY" scripts/build_track_r_v2_2_wavelengths.py \
    --protocol configs/track_r_v2_2/TRACK_R_v2_2.json \
    --data-dir "$DATA/visit_Bm_m3_trackr" \
    --output "$DATA/visit_Bm_m3_wavelengths.json" 2>&1 | tail -2
else
  echo "skip (exists)"
fi

echo "=== protocol ==="
if [ ! -f "$DATA/visit_Bm_protocol.json" ]; then
  "$PY" "$DATA/mimic_make_protocol_generic.py" \
    --wavelength "$DATA/visit_Bm_m3_wavelengths.json" \
    --out "$DATA/visit_Bm_protocol.json" --output-root "$DATA/results/visit_Bm" 2>&1 | tail -2
else
  echo "skip (exists)"
fi

echo "=== patch manifests ==="
cat > /tmp/patch_bm.py <<'PYEOF'
import json, sys
ROOT = "${MEDTRAJECTORY_ROOT}"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol
proto = load_track_r_protocol("${MEDTRAJECTORY_MIMIC_ROOT}"/visit_Bm_protocol.json)
for name in ("visit_Bm_m1_trackr", "visit_Bm_m2_trackr", "visit_Bm_m3_trackr"):
    p = f"${MEDTRAJECTORY_MIMIC_ROOT}"/{name}/prepare_manifest.json
    m = json.load(open(p, encoding="utf-8"))
    m["protocol_id"] = proto["data_protocol_id"]
    m["loss_contract"] = proto["loss_contract"]
    m["static_prefix"] = proto["static_prefix"]
    m["dynamic_bos"] = proto["dynamic_bos"]
    json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
    print("patched", name)
PYEOF
"$PY" /tmp/patch_bm.py

echo "=== CPU PHASE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# ---------------- wait for GPU to be fully free (A-seeds training AND its 3-seed eval) ----------------
waited=0
while pgrep -f train_car_rope_pretraining > /dev/null 2>&1 \
      || pgrep -f generation_eval > /dev/null 2>&1 \
      || [ ! -f "$DATA/results_visitA_3seeds.md" ]; do
  sleep 180
  waited=$((waited + 180))
  if [ "$waited" -gt 43200 ]; then echo "WARN: wait timeout (12h), proceeding"; break; fi
done
echo "=== GPU free, starting Bm training $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

launch () {
  local name="$1" ddir="$2"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/${ddir}_trackr" --track-r-protocol "$DATA/visit_Bm_protocol.json" \
    --include-static-prefix true --age-rope-variant legacy \
    --use-age-encoding true --use-age-rope false \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed 42 --no-tensorboard > "$LOGS/$name.log" 2>&1
}

launch bm_m1_a0 visit_Bm_m1 &
P1=$!
launch bm_m2_a0 visit_Bm_m2 &
P2=$!
launch bm_m3_a0 visit_Bm_m3 &
P3=$!
echo "launched Bm pids: $P1 $P2 $P3 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait $P1; wait $P2; wait $P3
echo "=== Bm training done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# ---------------- matched cohort + eval ----------------
echo "=== building matched cohort (M1-eligible test patients) ==="
cat > /tmp/mkcohort.py <<'PYEOF'
import sys
sys.path.insert(0, "${MEDTRAJECTORY_MIMIC_ROOT}")
from mimic_make_matched_cohort import eligible_eids
D = "${MEDTRAJECTORY_MIMIC_ROOT}"
e = eligible_eids(f"{D}/visit_Bm_m1_trackr", "test", 8, 3, 0.65, 10.0)
open(f"{D}/matched_bm_ukb.txt", "w").write("\n".join(sorted(e)) + "\n")
print("ukb matched cohort:", len(e))
e2 = eligible_eids(f"{D}/visit_Bm_m1_trackr", "test", 3, 2, 0.65, 10.0)
open(f"{D}/matched_bm_relaxed.txt", "w").write("\n".join(sorted(e2)) + "\n")
print("relaxed matched cohort:", len(e2))
PYEOF
"$PY" /tmp/mkcohort.py

run_mode () {
  local tag="$1" mh="$2" mf="$3" cap="$4" eidfile="$5"
  local out="$DATA/eval_bm_$tag"
  mkdir -p "$out"
  echo "==== bm mode $tag start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  local pids=()
  for spec in "bm_m1_a0:visit_Bm_m1" "bm_m2_a0:visit_Bm_m2" "bm_m3_a0:visit_Bm_m3"; do
    local name="${spec%%:*}" ddir="${spec##*:}"
    ( "$PY" "$DATA/mimic_generation_eval_trackr.py" \
        --ckpt "$DATA/runs/$name/checkpoints/last.pt" \
        --data-dir "$DATA/${ddir}_trackr" --split test --device cuda \
        --min-history-events "$mh" --min-future-events "$mf" \
        ${cap:+--max-patients "$cap"} --eid-file "$eidfile" \
        --out-dir "$out/$name" \
        --num-rollouts 20 --max-new-tokens 30 \
        --followup-years 10 --baseline-fraction 0.65 \
        > "$out/$name.stdout.log" 2>&1 ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "==== bm mode $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

run_mode ukb 8 3 "" "$DATA/matched_bm_ukb.txt"
run_mode relaxed 3 2 2000 "$DATA/matched_bm_relaxed.txt"
echo "=== BM EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
