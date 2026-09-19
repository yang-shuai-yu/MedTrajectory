#!/usr/bin/env bash
set -euo pipefail

session="carope_capacity_finalize"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

tmux has-session -t "${session}" 2>/dev/null && { echo "tmux session exists: ${session}" >&2; exit 2; }
tmux new-session -d -s "${session}" \
  "cd '${project_root}' && bash scripts/finalize_car_rope_capacity_comparison.sh"
echo "session=${session}"
