#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="${MEDTRAJECTORY_PYTHON:-python3}"
status_path="$repo_dir/results/track_g_v1/locked_test_queue_status.json"
log_path="$repo_dir/results/track_g_v1/locked_test_queue.log"
session_name=track_g_locked_test

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 2
fi

printf -v command 'cd %q && stage=locked-test; trap '\''code=$?; printf "{\\"status\\":\\"failed\\",\\"exit_code\\":%%s,\\"updated_at\\":\\"%%s\\"}\\n" "$code" "$(date --iso-8601=seconds)" > %q; exit "$code"'\'' ERR; printf "{\\"status\\":\\"running\\",\\"exit_code\\":null,\\"updated_at\\":\\"%%s\\"}\\n" "$(date --iso-8601=seconds)" > %q; %q -u scripts/run_track_g_locked_test.py --device cuda --execute; printf "{\\"status\\":\\"finished\\",\\"exit_code\\":0,\\"updated_at\\":\\"%%s\\"}\\n" "$(date --iso-8601=seconds)" > %q' \
  "$repo_dir" "$status_path" "$status_path" "$python_bin" "$status_path"
tmux new-session -d -s "$session_name" "exec bash -lc $(printf %q "$command") >> $(printf %q "$log_path") 2>&1"
echo "started tmux session: $session_name"
