#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "${MEDTRAJECTORY_ROOT}"

PY=${PY:-"${MEDTRAJECTORY_PYTHON:-python3}"}
DEVICE=${DEVICE:-cuda}
GPU_ID=${GPU_ID:-0}
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

INIT_CKPT=${INIT_CKPT:-results/gated_rope_ablation/gate100/ckpt.pt}
TRAIN_ITERS=${TRAIN_ITERS:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-300}
EVAL_ITERS=${EVAL_ITERS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
BOOTSTRAP=${BOOTSTRAP:-500}

TRAIN_OUT=${TRAIN_OUT:-results/monotonic_horizon/monotonic_gated_rope_gate100}
EVAL_OUT=${EVAL_OUT:-results/locked_test_horizon_risk/monotonic_gated_rope_gate100}
STATE_OUT=${STATE_OUT:-results/age_state_evaluation/state_generation_monotonic_gated_rope_gate100}

mkdir -p results/monotonic_horizon results/locked_test_horizon_risk results/age_state_evaluation

if [[ -f "$TRAIN_OUT/ckpt.pt" && -f "$TRAIN_OUT/history.json" ]]; then
  echo "[skip train] monotonic gated RoPE"
else
  echo "[train] monotonic gated RoPE horizon head"
  "$PY" src/semantic_delphi_ukb/train_medtrajectory_horizon_risk.py \
    --device "$DEVICE" \
    --init-from-ckpt "$INIT_CKPT" \
    --max-patients 0 \
    --max-iters "$TRAIN_ITERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-iters "$EVAL_ITERS" \
    --batch-size "$BATCH_SIZE" \
    --block-size 128 \
    --learning-rate 3e-4 \
    --weight-decay 0.1 \
    --next-event-loss-weight 0.2 \
    --tte-loss-weight 0.0 \
    --horizon-risk-loss-weight 1.0 \
    --use-age-encoding true \
    --use-age-rope true \
    --age-rope-gate 1.0 \
    --monotonic-horizon-risk \
    --horizons 5,10 \
    --seed 42 \
    --out-dir "$TRAIN_OUT" 2>&1 | tee "${TRAIN_OUT}.log"
fi

if [[ -f "$EVAL_OUT/aggregate_metrics.csv" && -f "$EVAL_OUT/raw_predictions.csv" ]]; then
  echo "[skip eval] monotonic gated RoPE"
else
  echo "[eval] monotonic gated RoPE locked test"
  "$PY" src/semantic_delphi_ukb/evaluate_horizon_risk_locked_test.py \
    --model medtrajectory \
    --checkpoint "$TRAIN_OUT/ckpt.pt" \
    --split test \
    --eval-all \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --block-size 128 \
    --bootstrap "$BOOTSTRAP" \
    --calibration-bins 10 \
    --use-age-encoding true \
    --use-age-rope true \
    --age-rope-gate 1.0 \
    --monotonic-horizon-risk true \
    --out-dir "$EVAL_OUT" 2>&1 | tee "${EVAL_OUT}.log"
fi

if [[ -f "$STATE_OUT/state_generation_summary.csv" ]]; then
  echo "[skip next-event eval] monotonic gated RoPE"
else
  echo "[eval next-event full-space] monotonic gated RoPE"
  "$PY" src/semantic_delphi_ukb/evaluate_state_generation_locked_test.py \
    --medtrajectory-checkpoint "$TRAIN_OUT/ckpt.pt" \
    --medtrajectory-name "Monotonic Gated RoPE main" \
    --split test \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --block-size 128 \
    --out-dir "$STATE_OUT" 2>&1 | tee "${STATE_OUT}.log"
fi

"$PY" src/semantic_delphi_ukb/compare_age_state_risk_metrics.py --include-ml

echo "[done] monotonic horizon ablation"
