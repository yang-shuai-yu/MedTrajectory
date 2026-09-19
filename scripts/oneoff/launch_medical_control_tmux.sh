#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

session="${1:?session name is required}"
out_dir="${2:?output directory is required}"
checkpoint="${3:?checkpoint path is required}"
data_dir="${4:?data directory is required}"
landmark_seed="${5:-1337}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin=""${MEDTRAJECTORY_PYTHON:-python3}""
diseases_yaml="${project_root}/docs/selected_diseases.yaml"

mkdir -p "${out_dir}"
command="'${python_bin}' -u '${project_root}/src/semantic_delphi_ukb/evaluate_medical_control_tasks.py' --mode explicit --checkpoint '${checkpoint}' --data-dir '${data_dir}' --split val --diseases-yaml '${diseases_yaml}' --out-dir '${out_dir}' --device cuda --horizons 1,5,10 --age-groups 50,55,60,65,70,75 --seed '${landmark_seed}' > '${out_dir}/stdout.log' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "out_dir=${out_dir}"
