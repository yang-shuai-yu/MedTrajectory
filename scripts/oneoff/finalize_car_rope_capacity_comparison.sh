#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
capacity_root="${project_root}/results/paper_protocol_v2/car_rope_capacity_evaluation_20260812_r1"
validation_a1="${project_root}/results/paper_protocol_v2/car_rope_locked_validation_20260811_r1/A1_car_rope_trunk/rows.json"
out_root="${project_root}/results/paper_protocol_v2/car_rope_capacity_comparison_20260812_r1"
status="${out_root}/status.json"
log="${out_root}/stdout.log"

test ! -e "${out_root}" || { echo "comparison output exists: ${out_root}" >&2; exit 2; }
mkdir -p "${out_root}"

write_status() {
  local state="$1"
  local phase="$2"
  local exit_code="${3:-null}"
  local temp="${status}.tmp"
  printf '{"label":"exploratory_capacity_study","state":"%s","phase":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${state}" "${phase}" "${exit_code}" "$(date -u +%FT%TZ)" > "${temp}"
  mv -f "${temp}" "${status}"
}

wait_for_evaluations() {
  while true; do
    for variant in A1-M A1-L; do
      for split in val; do
        test -f "${capacity_root}/${variant}/${split}/summary.json" || { sleep 30; continue 3; }
      done
    done
    return 0
  done
}

run_comparison() {
  local split="$1"
  local baseline="$2"
  local out_dir="${out_root}/${split}"
  PYTHONPATH="${project_root}/src" "${python_bin}" -u \
    "${project_root}/src/semantic_delphi_ukb/compare_horizon_control_tasks.py" \
    --input "A1-S=${baseline}" \
    --input "A1-M=${capacity_root}/A1-M/${split}/rows.json" \
    --input "A1-L=${capacity_root}/A1-L/${split}/rows.json" \
    --out-dir "${out_dir}" --bootstrap 1000 --seed 42 --calibration-bins 10 --memory-efficient
}

write_status running waiting_for_evaluations
if {
  wait_for_evaluations
  test -f "${validation_a1}"
  write_status running validation_comparison
  run_comparison val "${validation_a1}"
} > "${log}" 2>&1; then
  write_status finished finished 0
else
  code=$?
  write_status failed failed "${code}"
  exit "${code}"
fi
