#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
run_id="${1:?locked-test run id is required}"
attempt="${2:-primary}"
validation_run_id="${CAROPE_VALIDATION_RUN_ID:-car_rope_locked_validation_20260811_r1}"
root="${project_root}/results/paper_protocol_v2/${run_id}"
validation_root="${project_root}/results/paper_protocol_v2/${validation_run_id}"
audit="${project_root}/results/paper_protocol_v2/split_audits/multitype.json"
status="${root}/finalize_${attempt}_status.json"
log="${root}/finalize_${attempt}.stdout.log"
variants=(A0_sincos_control A1_car_rope_trunk A2_relative_horizon_query A3_full_carope)
freeze="${root}/freeze_manifest.json"
verification="${root}/freeze_verification.json"
if [[ "${attempt}" != primary ]]; then
  freeze="${root}/freeze_manifest_${attempt}.json"
  verification="${root}/freeze_verification_${attempt}.json"
fi

write_status() {
  local state="$1"
  local exit_code="${2:-null}"
  local temp="${status}.tmp"
  printf '{"state":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${state}" "${exit_code}" "$(date -u +%FT%TZ)" > "${temp}"
  mv -f "${temp}" "${status}"
}

wait_for_models() {
  while true; do
    pending=0
    for variant in "${variants[@]}"; do
      model_status="${root}/run_status/${variant}.json"
      if [[ ! -f "${model_status}" ]]; then
        pending=1
        continue
      fi
      state="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "${model_status}")"
      if [[ "${state}" == failed ]]; then
        echo "model evaluation failed: ${variant}" >&2
        return 3
      fi
      if [[ "${state}" != finished ]]; then
        pending=1
      fi
    done
    [[ "${pending}" -eq 0 ]] && return 0
    sleep 30
  done
}

run_finalize() {
  wait_for_models
  for variant in "${variants[@]}"; do
    test -f "${root}/${variant}/rows.json"
    test -f "${root}/${variant}/summary.json"
  done
  PYTHONPATH="${project_root}/src" "${python_bin}" -u "${project_root}/scripts/run_locked_test_paper_protocol.py" \
    --freeze-manifest "${freeze}" \
    --freeze-verification "${verification}" \
    --split-audit "${audit}" \
    --consistency "${validation_root}/validation_consistency.json" \
    --require-consistency validation_consistency \
    --input "A0=${root}/A0_sincos_control/rows.json" \
    --input "A1=${root}/A1_car_rope_trunk/rows.json" \
    --input "A2=${root}/A2_relative_horizon_query/rows.json" \
    --input "A3=${root}/A3_full_carope/rows.json" \
    --out-dir "${root}" --landmark-manifest "${root}/test_shared_landmarks.json" \
    --bootstrap 1000 --seed 42 --memory-efficient --allow-exploratory-gate-override
}

if [[ -e "${status}" || -e "${log}" ]]; then
  echo "refusing to overwrite finalization artifacts" >&2
  exit 2
fi
write_status running
if run_finalize > "${log}" 2>&1; then
  write_status finished 0
else
  code=$?
  write_status failed "${code}"
  exit "${code}"
fi
