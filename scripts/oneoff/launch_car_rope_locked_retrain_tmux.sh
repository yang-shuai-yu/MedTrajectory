#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

session="${1:?session name is required}"
spec="${2:-configs/paper_protocol_v1/CARoPE_locked_retrain_v2.json}"
stage="${3:-all}"
device="${4:-cuda}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
log_dir="${project_root}/results/paper_protocol_v2/launcher_logs"
log_path="${log_dir}/${session}.log"

test -f "${project_root}/${spec}"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi
if [[ -e "${log_path}" ]]; then
  echo "launcher log already exists: ${log_path}" >&2
  exit 3
fi
if [[ "${device}" == cuda* ]]; then
  command -v nvidia-smi >/dev/null
  nvidia-smi
fi
mkdir -p "${log_dir}"

command="cd '${project_root}' && export PYTHONPATH=src && '${python_bin}' -u scripts/run_car_rope_locked_retrain.py --spec '${project_root}/${spec}' --stage '${stage}' --device '${device}' --validate-inputs --execute > '${log_path}' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "log=${log_path}"
