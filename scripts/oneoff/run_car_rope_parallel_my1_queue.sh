#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

parent_pid="${1:?stopped parent runner PID is required}"
log_path="${2:?queue log path is required}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
spec="${project_root}/configs/paper_protocol_v1/CARoPE_locked_retrain_v2_my1.json"
runner="${project_root}/scripts/run_car_rope_locked_retrain.py"
a0_status="${project_root}/results/paper_protocol_v2/car_rope_locked_retrain_20260811/seed42/A0_sincos_control/pretraining/status.json"

if [[ -e "${log_path}" ]]; then
  echo "queue log already exists: ${log_path}" >&2
  exit 3
fi
exec >"${log_path}" 2>&1

echo "waiting_for_A0_pretraining"
until grep -q '"status": "finished"' "${a0_status}"; do
  kill -0 "${parent_pid}"
  sleep 30
done

parent_state="$(ps -o stat= -p "${parent_pid}" | xargs)"
if [[ "${parent_state}" != T* ]]; then
  echo "expected stopped parent ${parent_pid}, got state ${parent_state}" >&2
  exit 2
fi
kill -KILL "${parent_pid}"

"${python_bin}" -u "${runner}" \
  --spec "${spec}" --variants A0_sincos_control --stage horizon \
  --device cuda --validate-inputs --execute

"${python_bin}" -u "${runner}" \
  --spec "${spec}" --variants A2_relative_horizon_query --stage all \
  --device cuda --validate-inputs --allow-existing-output-root --execute
