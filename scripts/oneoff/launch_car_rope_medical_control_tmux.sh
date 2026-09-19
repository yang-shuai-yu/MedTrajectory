#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="${MEDTRAJECTORY_ROOT}"
python_bin=""${MEDTRAJECTORY_PYTHON:-python3}""
data_dir="${project_root}/data/paper_protocol_v1/multitype"
landmark_manifest="${project_root}/results/paper_protocol_v1/cross_profile_topk_20260810/val_shared_landmarks.json"

session="${1:?session name is required}"
variant="${2:?variant name is required}"
checkpoint="${project_root}/results/paper_protocol_v1/car_rope_validation_20260810/${variant}/horizon/checkpoints/best_val_horizon_auc.pt"
out_dir="${project_root}/results/paper_protocol_v1/car_rope_validation_20260810/${variant}/medical_control_val_shared_landmarks"

test -f "${checkpoint}" || { echo "missing checkpoint: ${checkpoint}" >&2; exit 3; }
test -f "${landmark_manifest}" || { echo "missing validation landmark manifest: ${landmark_manifest}" >&2; exit 3; }
mkdir -p "${out_dir}"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "session already exists: ${session}" >&2
  exit 2
fi

command="cd '${project_root}' && export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 && '${python_bin}' -u src/semantic_delphi_ukb/evaluate_medical_control_tasks.py --mode explicit --checkpoint '${checkpoint}' --data-dir '${data_dir}' --split val --diseases-yaml '${project_root}/docs/selected_diseases.yaml' --out-dir '${out_dir}' --device cuda --horizons 1,5 --age-groups 50,55,60,65,70,75 --batch-size 128 --seed 1337 --landmark-manifest '${landmark_manifest}' > '${out_dir}/stdout.log' 2>&1"
tmux new-session -d -s "${session}" "${command}"

echo "started=${session}"
echo "out_dir=${out_dir}"
