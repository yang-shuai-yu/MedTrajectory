#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
data_dir="${MEDTRAJECTORY_DATA_ROOT}"/data/paper_protocol_v2_locked_test/multitype
train_root="${project_root}/results/paper_protocol_v2/car_rope_locked_retrain_20260811/seed42"
run_id="${1:?locked-test run id is required}"
host_label="${2:?host label is required}"
shift 2
if [[ "$#" -eq 0 ]]; then
  echo "at least one variant is required" >&2
  exit 2
fi

root="${project_root}/results/paper_protocol_v2/${run_id}"
freeze="${root}/freeze_manifest.json"
landmarks="${root}/test_shared_landmarks.json"
status_root="${root}/run_status"
command_root="${root}/commands"
host_verification="${root}/freeze_verification_${host_label}.json"
mkdir -p "${status_root}" "${command_root}"
test -f "${freeze}"
test -f "${landmarks}"
test ! -e "${host_verification}"

PYTHONPATH="${project_root}/src" "${python_bin}" "${project_root}/scripts/verify_paper_protocol_freeze_manifest.py" \
  --freeze-manifest "${freeze}" --out "${host_verification}"

checkpoint_for() {
  case "$1" in
    A0_sincos_control) printf '%s\n' "${train_root}/$1/horizon_retry1/checkpoints/best_val_horizon_auc.pt" ;;
    A1_car_rope_trunk|A2_relative_horizon_query|A3_full_carope)
      printf '%s\n' "${train_root}/$1/horizon/checkpoints/best_val_horizon_auc.pt" ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

write_status() {
  local variant="$1"
  local state="$2"
  local exit_code="${3:-null}"
  local target="${status_root}/${variant}.json"
  local temp="${target}.tmp"
  printf '{"variant":"%s","host":"%s","state":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${variant}" "${host_label}" "${state}" "${exit_code}" "$(date -u +%FT%TZ)" > "${temp}"
  mv -f "${temp}" "${target}"
}

for variant in "$@"; do
  checkpoint="$(checkpoint_for "${variant}")"
  out_dir="${root}/${variant}"
  command_file="${command_root}/${variant}.command"
  status_file="${status_root}/${variant}.json"
  for path in "${out_dir}" "${command_file}" "${status_file}"; do
    if [[ -e "${path}" ]]; then
      echo "refusing to overwrite locked-test artifact: ${path}" >&2
      exit 2
    fi
  done
  test -f "${checkpoint}"
  mkdir -p "${out_dir}"
  command=(
    "${python_bin}" -u "${project_root}/src/semantic_delphi_ukb/evaluate_medical_control_tasks.py"
    --mode explicit --checkpoint "${checkpoint}" --data-dir "${data_dir}" --split test
    --diseases-yaml "${project_root}/docs/selected_diseases.yaml" --out-dir "${out_dir}"
    --device cuda --horizons 1,5,10 --age-groups 50,55,60,65,70,75
    --batch-size 128 --seed 1337 --landmark-manifest "${landmarks}" --calibration-bins 10
  )
  printf '%q ' "${command[@]}" > "${command_file}"
  printf '\n' >> "${command_file}"
  write_status "${variant}" running
  if PYTHONPATH="${project_root}/src" CUDA_VISIBLE_DEVICES=0 "${command[@]}" > "${out_dir}/stdout.log" 2>&1; then
    write_status "${variant}" finished 0
  else
    code=$?
    write_status "${variant}" failed "${code}"
    exit "${code}"
  fi
done

