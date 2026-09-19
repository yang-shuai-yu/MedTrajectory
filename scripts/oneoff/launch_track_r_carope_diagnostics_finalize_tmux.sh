#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

session="${1:-track_r_carope_diagnostics_finalize}"
run_tag="${2:-seed42_diagnostic2}"
current_run_tag="${3:-seed42_retry1}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
log_dir="${project_root}/results/track_r_v2_1/launcher_logs"
log_path="${log_dir}/${session}.log"

tmux has-session -t "${session}" 2>/dev/null && { echo "tmux session already exists: ${session}" >&2; exit 3; }
test ! -e "${log_path}" || { echo "launcher log already exists: ${log_path}" >&2; exit 4; }
mkdir -p "${log_dir}"

command="cd '${project_root}' && export PYTHONPATH=src && '${python_bin}' -u scripts/finalize_track_r_carope_diagnostics.py --run-tag '${run_tag}' --current-run-tag '${current_run_tag}' --bootstrap 1000 --seed 42 --execute > '${log_path}' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "log=${log_path}"
