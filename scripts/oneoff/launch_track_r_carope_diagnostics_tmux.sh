#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

stage="${1:-all}"
session="${2:-track_r_carope_diagnostics}"
device="${3:-cuda}"
run_tag="${4:-seed42_diagnostic1}"
current_run_tag="${5:-seed42_retry1}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
log_dir="${project_root}/results/track_r_v2_1/launcher_logs"
log_path="${log_dir}/${session}.log"

case "${stage}" in checkpoint_drift|a0_no_static|legacy_sex_bos|post_trunk_static|legacy_contract_rebuild|all) ;; *) echo "unknown stage: ${stage}" >&2; exit 2 ;; esac
tmux has-session -t "${session}" 2>/dev/null && { echo "tmux session already exists: ${session}" >&2; exit 3; }
test ! -e "${log_path}" || { echo "launcher log already exists: ${log_path}" >&2; exit 4; }
mkdir -p "${log_dir}"

command="cd '${project_root}' && export PYTHONPATH=src && '${python_bin}' -u scripts/run_track_r_carope_diagnostics.py --stage '${stage}' --device '${device}' --run-tag '${run_tag}' --current-run-tag '${current_run_tag}' --execute > '${log_path}' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "log=${log_path}"
