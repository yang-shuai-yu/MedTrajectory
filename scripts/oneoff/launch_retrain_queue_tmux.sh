#!/usr/bin/env bash
set -euo pipefail

session="${1:?session name is required}"
matrix_root="${2:?matrix root is required}"
seeds="${3:?seed list is required}"
experiments="${4:?experiment list is required}"
data_root="${5:?paper data root is required}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${matrix_root}"
cat > "${matrix_root}/queue_config.json" <<EOF
{"session":"${session}","host":"$(hostname)","seeds":"${seeds}","experiments":"${experiments}","data_root":"${data_root}","device":"cuda","gpu_id":0}
EOF

command="cd '${project_root}' && export PAPER_DATA_ROOT='${data_root}' PAPER_SEEDS='${seeds}' PAPER_EXPERIMENTS='${experiments}' PAPER_DEVICE=cuda PAPER_GPU_ID=0 PYTHONPATH='${project_root}/src'; bash scripts/run_paper_protocol_matrix.sh '${matrix_root}' >> '${matrix_root}/stdout.log' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "matrix_root=${matrix_root}"
