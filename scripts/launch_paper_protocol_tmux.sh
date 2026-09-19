#!/usr/bin/env bash
set -euo pipefail

experiment_key="${1:?usage: launch_paper_protocol_tmux.sh P0|P1|P2|P3|P4 TRAINING_SEED [run_dir]}"
training_seed="${2:?training seed is required}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${3:-}"
session="paper_${experiment_key}_seed${training_seed}_$(date -u +%Y%m%d_%H%M%S)"
command="cd '${project_root}' && bash scripts/run_paper_protocol_experiment.sh '${experiment_key}' '${training_seed}'"
if [[ -n "${run_dir}" ]]; then
  command="${command} '${run_dir}'"
fi
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
[[ -n "${run_dir}" ]] && echo "run_dir=${run_dir}"
