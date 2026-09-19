#!/bin/bash
# Re-run of the B/C three-seed evaluation with the cohort-matching bug fixed and
# the evaluation pinned to the corrected eval script.
set -uo pipefail
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
cd "$DATA"
echo "=== eval re-run start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash "$DATA/mimic_run_eval_bc_seeds.sh"
echo "=== eval re-run done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
