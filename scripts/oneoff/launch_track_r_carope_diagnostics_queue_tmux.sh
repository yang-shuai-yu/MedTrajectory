#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

stage_csv="${1:?comma-separated stages are required}"
session="${2:?tmux session is required}"
device="${3:-cuda}"
run_tag="${4:-seed42_diagnostic2}"
current_run_tag="${5:-seed42_retry1}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
log_dir="${project_root}/results/track_r_v2_1/launcher_logs"
log_path="${log_dir}/${session}.log"

IFS=',' read -r -a stages <<< "${stage_csv}"
commands=()
for stage in "${stages[@]}"; do
  case "${stage}" in
    checkpoint_drift|a0_no_static|legacy_sex_bos|post_trunk_static|legacy_contract_rebuild) ;;
    *) echo "unknown stage: ${stage}" >&2; exit 2 ;;
  esac
  commands+=("'${python_bin}' -u scripts/run_track_r_carope_diagnostics.py --stage '${stage}' --device '${device}' --run-tag '${run_tag}' --current-run-tag '${current_run_tag}' --execute")
done

tmux has-session -t "${session}" 2>/dev/null && { echo "tmux session already exists: ${session}" >&2; exit 3; }
test ! -e "${log_path}" || { echo "launcher log already exists: ${log_path}" >&2; exit 4; }
mkdir -p "${log_dir}"
: > "${log_path}"

joined=""
for item in "${commands[@]}"; do
  if [[ -n "${joined}" ]]; then
    joined+=" && "
  fi
  joined+="${item}"
done
command="cd '${project_root}' && export PYTHONPATH=src && { ${joined}; } >> '${log_path}' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "stages=${stage_csv}"
echo "log=${log_path}"
