#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

repo_dir="${MEDTRAJECTORY_ROOT}"
python_bin="${MEDTRAJECTORY_PYTHON:-python3}"
status_path="$repo_dir/results/track_g_v1/evaluation_queue_status.json"
stage=starting

write_status() {
  local state=$1
  local exit_code=${2:-null}
  local tmp_path="${status_path}.tmp"
  printf '{"status":"%s","stage":"%s","exit_code":%s,"updated_at":"%s"}\n' \
    "$state" "$stage" "$exit_code" "$(date --iso-8601=seconds)" > "$tmp_path"
  mv "$tmp_path" "$status_path"
}

fail() {
  local exit_code=$?
  write_status failed "$exit_code"
  exit "$exit_code"
}
trap fail ERR

cd "$repo_dir"
write_status running

stage=sampler-grid
write_status running
"$python_bin" -u scripts/run_track_g_v1.py --lane sampler-grid --device cuda --execute

stage=sampler-select
write_status running
"$python_bin" -u scripts/run_track_g_v1.py --lane sampler-select --device cpu --execute

stage=validate
write_status running
"$python_bin" -u scripts/run_track_g_v1.py --lane validate --device cuda --execute

stage=finalize
write_status running
"$python_bin" -u scripts/run_track_g_v1.py --lane finalize --device cpu --execute

stage=finished
write_status finished 0
