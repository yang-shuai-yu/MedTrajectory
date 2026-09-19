#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

REPO_DIR="${MEDTRAJECTORY_EXTERNAL_ROOT}"
PYTHON_BIN=""${MEDTRAJECTORY_PYTHON:-python3}""
CONFIG_PATH="semantic_delphi_ukb/config_medtrajectory_exp2_modern_baseline.py"
SESSION_NAME="${1:-medtrajectory_exp2_modern_baseline}"
LOG_DIR="$REPO_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/${SESSION_NAME}_${STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

DEVICE="cpu"
GPU_REASON="nvidia-smi unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_USED_MEM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  GPU_PROCS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)"
  if [[ "${GPU_PROCS}" == "0" && -n "${GPU_USED_MEM}" && "${GPU_USED_MEM}" -le 256 ]]; then
    DEVICE="cuda"
    GPU_REASON="gpu appears idle"
  else
    GPU_REASON="gpu busy: used_mem=${GPU_USED_MEM:-unknown}MiB compute_procs=${GPU_PROCS}"
  fi
fi

CMD="PYTHONUNBUFFERED=1 $PYTHON_BIN train_modern_multitype.py $CONFIG_PATH --device='$DEVICE'"

echo "repo_dir=$REPO_DIR"
echo "session_name=$SESSION_NAME"
echo "device=$DEVICE"
echo "reason=$GPU_REASON"
echo "log_path=$LOG_PATH"
echo "command=$CMD"

tmux new-session -d -s "$SESSION_NAME" "cd '$REPO_DIR' && $CMD 2>&1 | tee '$LOG_PATH'"
echo "started tmux session: $SESSION_NAME"
