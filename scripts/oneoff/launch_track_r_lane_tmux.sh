#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

lane="${1:?lane is required: audit, build, transformers, baselines, or medbert}"
session="${2:-track_r_${lane}}"
device="${3:-cuda}"
run_tag="${4:-seed42}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
log_dir="${project_root}/results/track_r_v2_1/launcher_logs"
log_path="${log_dir}/${session}.log"

case "${lane}" in audit|build|transformers|baselines|medbert) ;; *) echo "unknown lane: ${lane}" >&2; exit 2 ;; esac
tmux has-session -t "${session}" 2>/dev/null && { echo "tmux session already exists: ${session}" >&2; exit 3; }
test ! -e "${log_path}" || { echo "launcher log already exists: ${log_path}" >&2; exit 4; }
mkdir -p "${log_dir}"

command="cd '${project_root}' && export PYTHONPATH=src && '${python_bin}' -u scripts/run_track_r_v2_1.py --lane '${lane}' --device '${device}' --run-tag '${run_tag}' --execute > '${log_path}' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "log=${log_path}"
