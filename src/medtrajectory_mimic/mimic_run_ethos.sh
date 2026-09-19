#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# ETHOS-Matched generative baseline on MIMIC Design A.
# Queued to run ONLY after the Design B-matched evaluation has finished.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_ethos
mkdir -p "$LOGS"

# ---------- wait until the Bm pipeline and all GPU work is done ----------
waited=0
while pgrep -f mimic_run_bm_pipeline > /dev/null 2>&1 \
      || pgrep -f train_car_rope_pretraining > /dev/null 2>&1 \
      || pgrep -f generation_eval > /dev/null 2>&1; do
  sleep 180
  waited=$((waited + 180))
  if [ "$waited" -gt 86400 ]; then echo "WARN: wait timeout (24h), proceeding"; break; fi
done
echo "=== GPU free, training ETHOS-Matched $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

"$PY" "$DATA/mimic_train_matched_baseline.py" \
  --model ethos_matched \
  --data-dir "$DATA/visit_A_trackr" \
  --track-r-protocol "$DATA/visit_A_protocol.json" \
  --diseases-yaml "$YAML" \
  --run-dir "$DATA/runs/ethos_matched_a" \
  --device cuda --max-iters 100000 --seed 42 --no-tensorboard > "$LOGS/ethos_matched_a.log" 2>&1
echo "=== ETHOS-Matched training done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# ---------- eval on the same Design A cohorts ----------
EVAL=$DATA/mimic_generation_eval_trackr.py
run_mode () {
  local tag="$1" mh="$2" mf="$3" cap="$4"
  local out="$DATA/eval_ethos_$tag"
  mkdir -p "$out"
  echo "==== ethos mode $tag start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  "$PY" "$EVAL" --ckpt "$DATA/runs/ethos_matched_a/checkpoints/last.pt" \
    --data-dir "$DATA/visit_A_trackr" --split test --device cuda \
    --min-history-events "$mh" --min-future-events "$mf" \
    ${cap:+--max-patients "$cap"} \
    --out-dir "$out/ethos_matched_a" \
    --num-rollouts 20 --max-new-tokens 30 \
    --followup-years 10 --baseline-fraction 0.65 \
    > "$out/ethos_matched_a.stdout.log" 2>&1
  echo "==== ethos mode $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}
run_mode ukb 8 3 ""
run_mode relaxed 3 2 2000

echo "=== writing ETHOS vs A0 vs A2 comparison ==="
"$PY" - <<'PYEOF' > "$DATA/results_ethos_vs_carope.md" 2>&1
import json, math
from pathlib import Path
from statistics import mean, pstdev
D = Path("${MEDTRAJECTORY_MIMIC_ROOT}")
KEYS = [("hit_at_1","Hit@1"),("hit_at_10","Hit@10"),("diagnosis_jaccard","Diag Jaccard"),
        ("diagnosis_recall","Diag Recall"),("first_event_time_mae_days","Time MAE (d)"),
        ("event_count_mae","Count MAE"),("sequence_edit_distance","Seq edit"),("death_brier","Death Brier")]
def load(p):
    p = Path(p)
    if not (p/"summary.json").exists(): return None
    m = json.loads((p/"summary.json").read_text())["metrics"]
    return m
print("# ETHOS-Matched generative baseline vs CARoPE A0/A2 (Design A, MIMIC)\n")
for tag, desc in (("ukb","UKB-aligned cohort (>=8 history, >=3 future)"),
                  ("relaxed","Relaxed cohort (>=3 history, >=2 future, first 2000)")):
    print(f"\n## {desc}\n")
    rows = {
        "A0 (absolute, CARoPE)": D/f"eval_visitA_seeds_{tag}/vA_a0",
        "A2 (additive_v2_2, CARoPE)": D/f"eval_visitA_seeds_{tag}/vA_a2",
        "ETHOS-Matched": D/f"eval_ethos_{tag}/ethos_matched_a",
    }
    print("| Model | n | " + " | ".join(l for _, l in KEYS) + " |")
    print("|---|---:" + "|---:"*len(KEYS) + "|")
    got = {}
    for label, p in rows.items():
        m = load(p)
        if m is None:
            print(f"| {label} | - | MISSING |"); continue
        got[label] = m
        cells = []
        for k, _ in KEYS:
            v = m.get(k)
            cells.append("NA" if v is None or not isinstance(v,(int,float)) or not math.isfinite(v) else f"{v:.4f}")
        print(f"| {label} | {m.get('patient_count')} | " + " | ".join(cells) + " |")
    if "A2 (additive_v2_2, CARoPE)" in got and "ETHOS-Matched" in got:
        print("\n**Key contrast the baseline is meant to test — does Relative beat a matched non-CARoPE baseline?**\n")
        print("| Metric | A2 | ETHOS-Matched | A2 − ETHOS |")
        print("|---|---:|---:|---:|")
        for k, lbl in KEYS:
            a = got["A2 (additive_v2_2, CARoPE)"].get(k); e = got["ETHOS-Matched"].get(k)
            if a is None or e is None or not (isinstance(a,(int,float)) and isinstance(e,(int,float))) or not (math.isfinite(a) and math.isfinite(e)):
                print(f"| {lbl} | NA | NA | NA |"); continue
            print(f"| {lbl} | {a:.4f} | {e:.4f} | {a-e:+.4f} |")
PYEOF
echo "=== ETHOS PIPELINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
