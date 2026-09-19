#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
matrix_root="${1:-${project_root}/results/paper_protocol_v1/matrix_${timestamp}}"
session="paper_matrix_${timestamp}"
mkdir -p "${matrix_root}"
command="cd '${project_root}' && bash scripts/run_paper_protocol_matrix.sh '${matrix_root}' >> '${matrix_root}/stdout.log' 2>&1"
tmux new-session -d -s "${session}" "${command}"
echo "session=${session}"
echo "matrix_root=${matrix_root}"
