#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "${MEDTRAJECTORY_ROOT}"

PY=${PY:-"${MEDTRAJECTORY_PYTHON:-python3}"}
DEVICE=${DEVICE:-cuda}
GPU_ID=${GPU_ID:-0}
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

NO_ROPE_CKPT=${NO_ROPE_CKPT:-"${MEDTRAJECTORY_EXTERNAL_ROOT}"/ckpt/MedTrajectory_exp2_modern_no_rope/ckpt.pt}

TRAIN_ITERS=${TRAIN_ITERS:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-300}
EVAL_ITERS=${EVAL_ITERS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
BOOTSTRAP=${BOOTSTRAP:-500}

mkdir -p results/gated_rope_ablation results/locked_test_horizon_risk

train_gate() {
  local name="$1"
  local gate="$2"
  local out="results/gated_rope_ablation/${name}"
  if [[ -f "$out/ckpt.pt" && -f "$out/history.json" ]]; then
    echo "[skip train] $name"
  else
    echo "[train gated RoPE] $name gate=$gate"
    "$PY" src/semantic_delphi_ukb/train_medtrajectory_horizon_risk.py \
      --device "$DEVICE" \
      --init-from-ckpt "$NO_ROPE_CKPT" \
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
      --age-rope-gate "$gate" \
      --horizons 5,10 \
      --seed 42 \
      --out-dir "$out" 2>&1 | tee "${out}.log"
  fi
}

eval_gate() {
  local name="$1"
  local gate="$2"
  local out="results/locked_test_horizon_risk/gated_rope_${name}"
  if [[ -f "$out/aggregate_metrics.csv" && -f "$out/raw_predictions.csv" ]]; then
    echo "[skip eval] $name"
  else
    echo "[eval gated RoPE] $name gate=$gate"
    "$PY" src/semantic_delphi_ukb/evaluate_horizon_risk_locked_test.py \
      --model medtrajectory \
      --checkpoint "results/gated_rope_ablation/${name}/ckpt.pt" \
      --split test \
      --eval-all \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --block-size 128 \
      --bootstrap "$BOOTSTRAP" \
      --calibration-bins 10 \
      --use-age-encoding true \
      --use-age-rope true \
      --age-rope-gate "$gate" \
      --out-dir "$out" 2>&1 | tee "${out}.log"
  fi
}

train_gate "gate025" "0.25"
eval_gate "gate025" "0.25"

train_gate "gate050" "0.5"
eval_gate "gate050" "0.5"

train_gate "gate100" "1.0"
eval_gate "gate100" "1.0"

echo "[done] gated continuous-age RoPE ablation"
