#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
matrix_root="${1:-${project_root}/results/paper_protocol_v1/matrix_$(date -u +%Y%m%d_%H%M%S)}"
seeds="${PAPER_SEEDS:-42 43 44}"
experiments="${PAPER_EXPERIMENTS:-P0 P1 P2 P3 P4}"
mkdir -p "${matrix_root}"

write_status() {
  local state="$1" experiment="${2:-}" seed="${3:-}" exit_code="${4:-null}"
  printf '{"state":"%s","experiment":"%s","seed":"%s","exit_code":%s,"updated_at":"%s"}\n' \
    "${state}" "${experiment}" "${seed}" "${exit_code}" "$(date -Iseconds)" > "${matrix_root}/status.json.tmp"
  mv "${matrix_root}/status.json.tmp" "${matrix_root}/status.json"
}

if [[ "${PAPER_DEVICE:-auto}" == "auto" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  write_status waiting_gpu
  while true; do
    used_memory="$(nvidia-smi -i "${PAPER_GPU_ID:-0}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used_memory}" =~ ^[0-9]+$ ]] && (( used_memory < 1024 )); then
      export PAPER_DEVICE=cuda
      break
    fi
    sleep 60
  done
fi

write_status running
for seed in ${seeds}; do
  for experiment in ${experiments}; do
    run_dir="${matrix_root}/${experiment}_seed${seed}"
    write_status running "${experiment}" "${seed}"
    set +e
    bash "${project_root}/scripts/run_paper_protocol_experiment.sh" "${experiment}" "${seed}" "${run_dir}"
    code=$?
    set -e
    if [[ ${code} -ne 0 ]]; then
      write_status failed "${experiment}" "${seed}" "${code}"
      exit "${code}"
    fi
  done
done
write_status finished
