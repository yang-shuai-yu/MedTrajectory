#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

PYTHON_BIN="${PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
PROJECT_ROOT="${PROJECT_ROOT:-"${MEDTRAJECTORY_ROOT}"}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:${PYTHONPATH:-}"

"$PYTHON_BIN" src/semantic_delphi_ukb/train_architecture_baselines.py \
  --model bert \
  --device cpu \
  --max-patients 512 \
  --max-iters 50 \
  --eval-interval 25 \
  --eval-iters 2 \
  --batch-size 16 \
  --out-dir results/architecture_baselines/bert_smoke

if "$PYTHON_BIN" - <<'PY'
try:
    import mamba_ssm  # noqa: F401
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
  "$PYTHON_BIN" src/semantic_delphi_ukb/train_architecture_baselines.py \
    --model mamba \
    --device cpu \
    --max-patients 512 \
    --max-iters 50 \
    --eval-interval 25 \
    --eval-iters 2 \
    --batch-size 16 \
    --out-dir results/architecture_baselines/mamba_smoke
else
  echo "mamba_ssm is not installed; skipping Mamba smoke run."
fi
