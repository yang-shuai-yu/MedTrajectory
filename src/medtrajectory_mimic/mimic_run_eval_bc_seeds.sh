#!/bin/bash
# Evaluate the 3-seed runs for Bm-matched multitype (Exp1) and Design B / C Experiment 2,
# then print the 3-seed summaries. Runs after mimic_run_bc_seeds.sh completes.
#
# 2026-09-13: single-threaded (OMP_NUM_THREADS=1) and 6-way concurrent.  The
# previous settings inherited PyTorch's 104-thread default per process, which
# burned ~20x the necessary CPU in thread-pool spin and throttled the cgroup
# (93.7% of periods).  Training is ~5x faster with the fix; the same applies here.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
EVAL=$DATA/mimic_generation_eval_trackr.py
export PYTHONPATH="$REPO:$REPO/src:$REPO/scripts"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# 2026-09-13: concurrency raised 6 -> 14.
# Measured with 6 concurrent evals: each process pinned at exactly 100% CPU
# (i.e. ONE core -- OMP_NUM_THREADS=1, and the rollout loop is single-threaded
# Python), 6 of the 20-core cgroup quota in use, GPU memory-controller
# utilisation 5%, 5.2 GB of 49 GB VRAM.  nvidia-smi's utilization.gpu reads 99%
# because tiny kernels are almost always resident, which is not a measure of how
# full the device is.  The evaluation is therefore Python-loop-bound, not
# GPU-throughput-bound, and 14 of 20 cores were idle.
CONC=${CONC:-6}
OUT=$DATA/results_bc_3seeds.md

ev () {  # ev <out_tag> <model> <ddir> <mh> <mf> <cap> <eidfile>
  local tag="$1" name="$2" ddir="$3" mh="$4" mf="$5" cap="$6" eid="$7"
  local dir="$DATA/$tag/$name"
  [ -f "$dir/summary.json" ] && { echo "skip $tag/$name (done)"; return; }
  mkdir -p "$dir"
  echo "  [$(date -u +%H:%M:%SZ)] eval $tag/$name"
  "$PY" "$EVAL" --ckpt "$DATA/runs/$name/checkpoints/last.pt" \
    --data-dir "$DATA/$ddir" --split test --device cuda \
    --min-history-events "$mh" --min-future-events "$mf" \
    ${cap:+--max-patients "$cap"} ${eid:+--eid-file "$eid"} \
    --out-dir "$dir" --num-rollouts 20 --max-new-tokens 30 \
    --followup-years 10 --baseline-fraction 0.65 > "$dir.stdout.log" 2>&1
  echo "  [$(date -u +%H:%M:%SZ)] done $tag/$name exit=$?"
}

mode () {  # mode <tag_suffix> <mh> <mf> <cap> <eidfile>
  local suf="$1" mh="$2" mf="$3" cap="$4" eid="$5"
  echo "==== $suf start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  local pids=()
  # Bm matched multitype
  for seed in 42 43 44; do
    for m in m1 m2 m3; do
      local n="bm_${m}_a0"; [ "$seed" = "42" ] || n="bm_${m}_a0_s${seed}"
      ev "eval_bm3_$suf" "$n" "visit_Bm_${m}_trackr" "$mh" "$mf" "$cap" "$eid" &
      pids+=($!); [ "${#pids[@]}" -ge $CONC ] && { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }
    done
  done
  # Design B / C Experiment 2
  for seed in 42 43 44; do
    local s0="vB_m3_a0" s2="vB_m3_a2"
    [ "$seed" = "42" ] || { s0="vB_m3_a0_s${seed}"; s2="vB_m3_a2_s${seed}"; }
    ev "eval_B3_$suf" "$s0" "visit_B_m3_trackr" "$mh" "$mf" "$cap" "" &
    pids+=($!); [ "${#pids[@]}" -ge $CONC ] && { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }
    ev "eval_B3_$suf" "$s2" "visit_B_m3_trackr" "$mh" "$mf" "$cap" "" &
    pids+=($!); [ "${#pids[@]}" -ge $CONC ] && { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }
    local c0="vC_a0" c2="vC_a2"
    [ "$seed" = "42" ] || { c0="vC_a0_s${seed}"; c2="vC_a2_s${seed}"; }
    ev "eval_C3_$suf" "$c0" "visit_C_trackr" "$mh" "$mf" "$cap" "" &
    pids+=($!); [ "${#pids[@]}" -ge $CONC ] && { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }
    ev "eval_C3_$suf" "$c2" "visit_C_trackr" "$mh" "$mf" "$cap" "" &
    pids+=($!); [ "${#pids[@]}" -ge $CONC ] && { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "==== $suf done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

# 2026-09-13 FIX: the Bm "matched" multitype comparison MUST pass --eid-file.
# Without it the eligibility filter (>=8 history + >=3 future events) independently
# selects 541 participants for M1 (diagnosis only) but 837 for M2/M3 (which also
# admit procedures).  The extra 296 participants have more events by construction,
# so the comparison is confounded in the direction that favours multitype, and the
# unequal-n run must never be reported under a "matched" label.
mode ukb 8 3 "" "$DATA/matched_bm_ukb.txt"
mode relaxed 3 2 2000 "$DATA/matched_bm_relaxed.txt"

echo "=== verifying cohort matching $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if ! "$PY" "$DATA/verify_eval_cohorts.py"; then
  echo "!!! COHORT MATCH VERIFICATION FAILED -- do NOT use these summaries"
  echo "!!! results_bc_3seeds.md was NOT written"
  exit 1
fi

echo "=== writing 3-seed summaries $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
{
  echo "# Three-seed summaries for Bm-matched multitype and Design B/C temporal comparisons"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## Experiment 2 â?Design B (visit level, matched A0-vs-A2 on the same representation)"
  echo
  for suf in ukb relaxed; do
    echo "### cohort: $suf"
    echo
    "$PY" "$DATA/mimic_analyze_3seed_generic.py" --tag "eval_B3_$suf" \
      --a0 "vB_m3_a0,vB_m3_a0_s43,vB_m3_a0_s44" \
      --a2 "vB_m3_a2,vB_m3_a2_s43,vB_m3_a2_s44" \
      --label "Design B Experiment 2 ($suf)"
  done
  echo "## Experiment 2 â?Design C (visit level, composite tokens)"
  echo
  for suf in ukb relaxed; do
    echo "### cohort: $suf"
    echo
    "$PY" "$DATA/mimic_analyze_3seed_generic.py" --tag "eval_C3_$suf" \
      --a0 "vC_a0,vC_a0_s43,vC_a0_s44" \
      --a2 "vC_a2,vC_a2_s43,vC_a2_s44" \
      --label "Design C Experiment 2 ($suf)"
  done
  echo "## Experiment 1 â?Bm matched multitype (M1 / M2 / M3), 3-seed means"
  echo
  for suf in ukb relaxed; do
    echo "### cohort: $suf"
    echo
    "$PY" "$DATA/mimic_analyze_bm_multitype.py" --tag "eval_bm3_$suf" --label "Bm matched multitype ($suf)"
  done
} > "$OUT" 2>&1
echo "=== wrote $OUT $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
