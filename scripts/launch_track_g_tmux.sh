#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 SESSION LANE DEVICE [run_track_g_v1.py arguments...]" >&2
  exit 2
fi

session_name=$1
lane=$2
device=$3
shift 3

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command=(python -u scripts/run_track_g_v1.py --lane "$lane" --device "$device" --execute "$@")
printf -v rendered '%q ' "${command[@]}"
printf -v shell_command 'cd %q && %s' "$repo_dir" "$rendered"
printf -v tmux_command 'exec bash -lc %q' "$shell_command"

tmux new-session -d -s "$session_name" "$tmux_command"
echo "started tmux session: $session_name"
