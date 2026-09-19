#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

repo_dir="${MEDTRAJECTORY_ROOT}"
python_bin="${MEDTRAJECTORY_PYTHON:-python3}"
cd "$repo_dir"

active_sessions=(track_g_e42 track_g_f42 track_g_e43 track_g_f43)
first_batch_status=(
  results/track_g_v1/runs/seed42/pretraining/ETHOS-Matched/status.json
  results/track_g_v1/runs/seed42/pretraining/Foresight-Matched/status.json
  results/track_g_v1/runs/seed43/pretraining/ETHOS-Matched/status.json
  results/track_g_v1/runs/seed43/pretraining/Foresight-Matched/status.json
)
seed44_status=(
  results/track_g_v1/runs/seed44/pretraining/ETHOS-Matched/status.json
  results/track_g_v1/runs/seed44/pretraining/Foresight-Matched/status.json
)

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

require_finished() {
  "$python_bin" - "$@" <<'PY'
import json
import sys
from pathlib import Path

failed = []
for value in sys.argv[1:]:
    path = Path(value)
    status = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if status.get("status") != "finished":
        failed.append(f"{path}: {status.get('status', 'missing')}")
if failed:
    raise SystemExit("training gate failed: " + "; ".join(failed))
PY
}

log "waiting for the first four Track G training sessions"
while :; do
  active=0
  for session in "${active_sessions[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      active=1
    fi
  done
  if [[ $active -eq 0 ]]; then
    break
  fi
  sleep 60
done

require_finished "${first_batch_status[@]}"
log "first four training runs finished; starting both seed 44 runs"

set +e
"$python_bin" -u scripts/run_track_g_v1.py --lane train --device cuda --execute \
  --pair 44:ETHOS-Matched &
ethos_pid=$!
"$python_bin" -u scripts/run_track_g_v1.py --lane train --device cuda --execute \
  --pair 44:Foresight-Matched &
foresight_pid=$!
wait "$ethos_pid"
ethos_rc=$?
wait "$foresight_pid"
foresight_rc=$?
set -e

if [[ $ethos_rc -ne 0 || $foresight_rc -ne 0 ]]; then
  log "seed 44 training failed: ETHOS=$ethos_rc Foresight=$foresight_rc"
  exit 1
fi
require_finished "${seed44_status[@]}"

log "all six training runs finished; starting sampler grid"
"$python_bin" -u scripts/run_track_g_v1.py --lane sampler-grid --device cuda --execute
log "sampler grid finished; selecting family samplers"
"$python_bin" -u scripts/run_track_g_v1.py --lane sampler-select --device cpu --execute
log "sampler selection finished; starting validation"
"$python_bin" -u scripts/run_track_g_v1.py --lane validate --device cuda --execute
log "validation finished; creating final assessment"
"$python_bin" -u scripts/run_track_g_v1.py --lane finalize --device cpu --execute
log "Track G validation pipeline finished; locked test remains disabled"
