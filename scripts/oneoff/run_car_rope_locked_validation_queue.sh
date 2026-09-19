#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
data_dir="${MEDTRAJECTORY_DATA_ROOT}"/data/paper_protocol_v2_locked_test/multitype
train_root="${project_root}/results/paper_protocol_v2/car_rope_locked_retrain_20260811/seed42"
validation_run_id="${CAROPE_VALIDATION_RUN_ID:-car_rope_locked_validation_20260811}"
out_root="${project_root}/results/paper_protocol_v2/${validation_run_id}"
landmark_manifest="${out_root}/val_shared_landmarks.json"
status_root="${out_root}/run_status"
command_root="${out_root}/commands"

host_label="${1:?host label is required}"
shift
if [[ "$#" -eq 0 ]]; then
  echo "at least one variant is required" >&2
  exit 2
fi

mkdir -p "${status_root}" "${command_root}"
test -f "${landmark_manifest}" || { echo "missing validation landmark manifest: ${landmark_manifest}" >&2; exit 3; }

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
  out_dir="${out_root}/${variant}"
  stdout_log="${out_dir}/stdout.log"
  command_file="${command_root}/${variant}.command"
  test -f "${checkpoint}" || { echo "missing checkpoint: ${checkpoint}" >&2; exit 3; }
  if [[ -e "${out_dir}" || -e "${command_file}" || -e "${status_root}/${variant}.json" ]]; then
    echo "refusing to overwrite existing validation output for ${variant}" >&2
    exit 2
  fi
  mkdir -p "${out_dir}"
  command=(
    "${python_bin}" -u "${project_root}/src/semantic_delphi_ukb/evaluate_medical_control_tasks.py"
    --mode explicit
    --checkpoint "${checkpoint}"
    --data-dir "${data_dir}"
    --split val
    --diseases-yaml "${project_root}/docs/selected_diseases.yaml"
    --out-dir "${out_dir}"
    --device cuda
    --horizons 1,5,10
    --age-groups 50,55,60,65,70,75
    --batch-size 128
    --seed 1337
    --landmark-manifest "${landmark_manifest}"
    --calibration-bins 10
  )
  printf '%q ' "${command[@]}" > "${command_file}"
  printf '\n' >> "${command_file}"
  write_status "${variant}" running
  if PYTHONPATH="${project_root}/src" CUDA_VISIBLE_DEVICES=0 "${command[@]}" > "${stdout_log}" 2>&1; then
    write_status "${variant}" finished 0
  else
    code=$?
    write_status "${variant}" failed "${code}"
    exit "${code}"
  fi
done
