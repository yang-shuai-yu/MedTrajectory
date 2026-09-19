#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

variant="${1:?variant is required: A1-M or A1-L}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
data_dir="${MEDTRAJECTORY_DATA_ROOT}"/data/paper_protocol_v2_locked_test/multitype
train_root="${project_root}/results/paper_protocol_v2/car_rope_capacity_scaling_20260812/seed42"
eval_root="${project_root}/results/paper_protocol_v2/car_rope_capacity_evaluation_20260812_r1/${variant}"
checkpoint="${train_root}/${variant}/horizon/checkpoints/best_val_horizon_auc.pt"
existing_root="${project_root}/results/paper_protocol_v2"

case "${variant}" in A1-M|A1-L) ;; *) echo "unknown capacity variant: ${variant}" >&2; exit 2 ;; esac
test -f "${checkpoint}"

mkdir -p "${eval_root}"
printf '{"label":"exploratory_capacity_study","checkpoint_selection_split":"val","test_used_for_selection":false}\n' \
  > "${eval_root}/study_label.json"

for split in val; do
  out_dir="${eval_root}/${split}"
  test ! -e "${out_dir}" || { echo "evaluation output exists: ${out_dir}" >&2; exit 3; }
  command=(
    "${python_bin}" -u "${project_root}/src/semantic_delphi_ukb/evaluate_medical_control_tasks.py"
    --mode explicit --checkpoint "${checkpoint}" --data-dir "${data_dir}" --split "${split}"
    --diseases-yaml "${project_root}/docs/selected_diseases.yaml" --out-dir "${out_dir}"
    --device cuda --horizons 1,5,10 --age-groups 50,55,60,65,70,75
    --batch-size 128 --seed 1337 --calibration-bins 10
  )
  landmark_manifest=""
  case "${split}" in
    val) landmark_manifest="${existing_root}/car_rope_locked_validation_20260811_r1/val_shared_landmarks.json" ;;
  esac
  if [[ -n "${landmark_manifest}" ]]; then
    test -f "${landmark_manifest}"
    command+=(--landmark-manifest "${landmark_manifest}")
  fi
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${project_root}/src" "${command[@]}" > "${out_dir}/stdout.log" 2>&1
done
