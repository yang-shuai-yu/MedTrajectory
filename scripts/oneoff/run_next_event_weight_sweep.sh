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
WEIGHTS=${WEIGHTS:-"0.2 0.5 1.0"}

mkdir -p results/next_event_weight_sweep results/locked_test_horizon_risk results/age_state_evaluation

weight_tag() {
  local value="$1"
  printf "%s" "$value" | sed 's/\./p/g'
}

for next_w in $WEIGHTS; do
  tag="$(weight_tag "$next_w")"
  train_out="results/next_event_weight_sweep/monotonic_gated_rope_nextw${tag}"
  eval_out="results/locked_test_horizon_risk/monotonic_gated_rope_nextw${tag}"
  state_out="results/age_state_evaluation/state_generation_monotonic_nextw${tag}"

  if [[ -f "$train_out/ckpt.pt" && -f "$train_out/history.json" ]]; then
    echo "[skip train] next_event_loss_weight=$next_w"
  else
    echo "[train] monotonic gated RoPE next_event_loss_weight=$next_w"
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
      --next-event-loss-weight "$next_w" \
      --tte-loss-weight 0.0 \
      --horizon-risk-loss-weight 1.0 \
      --use-age-encoding true \
      --use-age-rope true \
      --age-rope-gate 1.0 \
      --monotonic-horizon-risk \
      --horizons 5,10 \
      --seed 42 \
      --out-dir "$train_out" 2>&1 | tee "${train_out}.log"
  fi

  if [[ -f "$eval_out/aggregate_metrics.csv" && -f "$eval_out/raw_predictions.csv" ]]; then
    echo "[skip horizon eval] next_event_loss_weight=$next_w"
  else
    echo "[eval horizon] next_event_loss_weight=$next_w"
    "$PY" src/semantic_delphi_ukb/evaluate_horizon_risk_locked_test.py \
      --model medtrajectory \
      --checkpoint "$train_out/ckpt.pt" \
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
      --out-dir "$eval_out" 2>&1 | tee "${eval_out}.log"
  fi

  if [[ -f "$state_out/state_generation_summary.csv" ]]; then
    echo "[skip next-event eval] next_event_loss_weight=$next_w"
  else
    echo "[eval next-event full-space] next_event_loss_weight=$next_w"
    "$PY" src/semantic_delphi_ukb/evaluate_state_generation_locked_test.py \
      --medtrajectory-checkpoint "$train_out/ckpt.pt" \
      --medtrajectory-name "Monotonic Gated RoPE nextw=$next_w" \
      --split test \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --block-size 128 \
      --out-dir "$state_out" 2>&1 | tee "${state_out}.log"
  fi
done

"$PY" src/semantic_delphi_ukb/compare_age_state_risk_metrics.py --include-ml

echo "[done] next-event auxiliary weight sweep"
