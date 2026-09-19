#!/usr/bin/env bash
set -euo pipefail

root="${MEDTRAJECTORY_ROOT}"
run="$root/results/track_r_v2_1/runs/seed42_censorfix1"
status="$run/Cox-R/validation/status.json"
summary="$run/Cox-R/validation/summary.json"

while [[ ! -s "$summary" ]]; do
  if [[ -s "$status" ]] && grep -q '"status": "failed"' "$status"; then
    cat "$status"
    exit 1
  fi
  sleep 60
done

bash "$root/scripts/launch_track_r_censorfix1_compare_freeze.sh"
