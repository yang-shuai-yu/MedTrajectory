#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
validation_run_id="${CAROPE_VALIDATION_RUN_ID:-car_rope_locked_validation_20260811}"
validation_root="${project_root}/results/paper_protocol_v2/${validation_run_id}"
out_dir="${validation_root}/validation_comparison"
status_path="${validation_root}/compare_status.json"
stdout_log="${validation_root}/compare.stdout.log"

for path in "${out_dir}" "${status_path}" "${stdout_log}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing path: ${path}" >&2
    exit 2
  fi
done

write_status() {
  local state="$1"
  local exit_code="${2:-null}"
  local temp="${status_path}.tmp"
  printf '{"state":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${state}" "${exit_code}" "$(date -u +%FT%TZ)" > "${temp}"
  mv -f "${temp}" "${status_path}"
}

command=(
  "${python_bin}" -u "${project_root}/src/semantic_delphi_ukb/compare_horizon_control_tasks.py"
  --input "A0=${validation_root}/A0_sincos_control/rows.json"
  --input "A1=${validation_root}/A1_car_rope_trunk/rows.json"
  --input "A2=${validation_root}/A2_relative_horizon_query/rows.json"
  --input "A3=${validation_root}/A3_full_carope/rows.json"
  --out-dir "${out_dir}"
  --bootstrap 1000
  --seed 42
  --calibration-bins 10
  --memory-efficient
)

write_status running
if PYTHONPATH="${project_root}/src" "${command[@]}" > "${stdout_log}" 2>&1; then
  write_status finished 0
else
  code=$?
  write_status failed "${code}"
  exit "${code}"
fi
