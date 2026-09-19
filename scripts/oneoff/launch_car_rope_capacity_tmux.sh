#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

variant="${1:?variant is required: A1-M or A1-L}"
session="${2:-carope_capacity_${variant}}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
spec="${project_root}/configs/paper_protocol_v1/CARoPE_capacity_scaling_v1.json"
log_dir="${project_root}/results/paper_protocol_v2/launcher_logs"
log_path="${log_dir}/${session}.log"

case "${variant}" in A1-M|A1-L) ;; *) echo "unknown capacity variant: ${variant}" >&2; exit 2 ;; esac
tmux has-session -t "${session}" 2>/dev/null && { echo "tmux session exists: ${session}" >&2; exit 2; }
test ! -e "${log_path}" || { echo "launcher log exists: ${log_path}" >&2; exit 3; }
nvidia-smi
mkdir -p "${log_dir}"

command="cd '${project_root}' && export PYTHONPATH=src && export CUDA_VISIBLE_DEVICES=0 && '${python_bin}' -u scripts/run_car_rope_capacity_scaling.py --spec '${spec}' --stage all --device cuda --variants '${variant}' --validate-inputs --execute > '${log_path}' 2>&1 && bash scripts/evaluate_car_rope_capacity.sh '${variant}' >> '${log_path}' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "log=${log_path}"
