#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Regenerate results_bc_3seeds.md with the FINAL, internally consistent analysis.
#
# Why this exists: the in-flight driver (mimic_run_eval_bc_seeds.sh) writes
# results_bc_3seeds.md using mimic_analyze_3seed_generic.py and
# mimic_analyze_bm_multitype.py.  Those two report three-seed means with
# pstdev (ddof = 0) and perform NO paired inference, so their numbers would
# contradict the paired analysis used in the technical report and the manuscript
# draft (which use the sample SD, ddof = 1, and participant-level bootstrap CIs).
# This script re-writes the artifact from the paired analysis and keeps the
# descriptive summaries only as a clearly-labelled appendix.
#
# Run after mimic_rerun_eval_bc.sh has finished.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
REPO="${MEDTRAJECTORY_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
OUT=$DATA/results_bc_3seeds.md
cd "$DATA"
export OMP_NUM_THREADS=1

echo "=== final report $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# ---- 1. the cohort-match gate must pass before anything is reported ----
echo "[1/4] cohort-match gate"
DISCLOSURE=/tmp/cohort_disclosure.txt
: > "$DISCLOSURE"
"$PY" "$DATA/verify_eval_cohorts.py" > /tmp/cohort_gate.txt 2>&1
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then
  # A mismatch is fatal UNLESS it is a rounding-level shortfall caused by the
  # --max-patients cap being applied in each representation's own index order
  # (the relaxed Bm cohort).  The paired statistics then run on the common
  # subset, and the shortfall must be disclosed in the report.
  if "$PY" "$DATA/check_gate_tolerance.py" /tmp/cohort_gate.txt 0.99 > "$DISCLOSURE" 2>&1; then
    echo "    gate reported a mismatch; shortfall is within tolerance and will be DISCLOSED"
    cat "$DISCLOSURE" | sed 's/^/    /'
  else
    echo "!!! COHORT MATCH GATE FAILED beyond tolerance -- results_bc_3seeds.md NOT written"
    cat /tmp/cohort_gate.txt
    cat "$DISCLOSURE"
    exit 1
  fi
fi
if grep -q 'still pending' /tmp/cohort_gate.txt; then
  echo "    NOTE: some groups still pending; the report will mark them INCOMPLETE"
fi
tail -1 /tmp/cohort_gate.txt || true

# ---- 2. three-seed paired analysis (the authoritative numbers) ----
echo "[2/4] three-seed paired analysis"
"$PY" "$DATA/mimic_paired_3seed.py" > /tmp/paired_3seed.md 2>&1
"$PY" "$DATA/mimic_ethos_paired.py" > /tmp/paired_ethos.md 2>&1
"$PY" "$DATA/mimic_final_conclusions.py" > /tmp/final_conclusions.md 2>&1

# ---- 3. descriptive summaries (appendix only) ----
echo "[3/4] descriptive summaries (appendix)"
{
  "$PY" "$DATA/mimic_analyze_bm_multitype.py" --tag eval_bm3_ukb \
    --label "Bm matched multitype, UKB-aligned cohort"
  for suf in ukb relaxed; do
    "$PY" "$DATA/mimic_analyze_3seed_generic.py" --tag "eval_B3_$suf" \
      --a0 "vB_m3_a0,vB_m3_a0_s43,vB_m3_a0_s44" \
      --a2 "vB_m3_a2,vB_m3_a2_s43,vB_m3_a2_s44" --label "Design B ($suf)"
    "$PY" "$DATA/mimic_analyze_3seed_generic.py" --tag "eval_C3_$suf" \
      --a0 "vC_a0,vC_a0_s43,vC_a0_s44" \
      --a2 "vC_a2,vC_a2_s43,vC_a2_s44" --label "Design C ($suf)"
  done
} > /tmp/descriptive.md 2>&1

# ---- 4. assemble ----
echo "[4/4] assembling $OUT"
{
  echo "# MIMIC-IV cross-setting replication — three-seed results"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`mimic_final_report.sh\`."
  echo
  echo "> **The authoritative numbers are in Part A (paired analysis).** Part A uses participant-level"
  echo "> pairing, the sample SD across seeds (ddof = 1) and a 10,000-replicate participant bootstrap,"
  echo "> matching the conventions used for the UKB generation table. Part B is a descriptive appendix"
  echo "> that reports three-seed means with \`pstdev\` (ddof = 0) and no paired inference; it is included"
  echo "> only for continuity with earlier working notes and **must not be quoted**."
  echo
  echo "---"
  echo
  echo "## Part A — final conclusions (paired, three seeds)"
  echo
  if [ -s "$DISCLOSURE" ]; then
    echo "> **DISCLOSED COHORT SHORTFALL.** The cohort-match gate reported a mismatch:"
    echo ">"
    sed 's/^/> /' "$DISCLOSURE"
    echo ">"
    echo "> Cause: the relaxed Bm cohort uses \`--eid-file matched_bm_relaxed.txt\` with"
    echo "> \`--max-patients 2000\`, and the cap is applied in each representation's own"
    echo "> \`patient_index\` order. Because the three representations order participants"
    echo "> differently, the cap lands on a slightly different participant in each arm."
    echo "> Every relaxed statistic below is therefore computed on the COMMON SUBSET,"
    echo "> whose size is printed as \`n\` with each contrast. The UKB-aligned cohort is"
    echo "> not affected (541 < the cap, so the truncation never binds) and is exactly paired."
    echo
  fi
  cat /tmp/final_conclusions.md
  echo
  echo "---"
  echo
  echo "## Part A2 — full paired tables"
  echo
  cat /tmp/paired_3seed.md
  echo
  cat /tmp/paired_ethos.md
  echo
  echo "---"
  echo
  echo "## Part B — descriptive appendix (ddof = 0, unpaired; do not quote)"
  echo
  cat /tmp/descriptive.md
} > "$OUT"

echo "=== wrote $OUT ($(wc -l < "$OUT") lines) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
