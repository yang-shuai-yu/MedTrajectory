#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "${MEDTRAJECTORY_ROOT}"

PY=${PY:-"${MEDTRAJECTORY_PYTHON:-python3}"}
DEVICE=${DEVICE:-cuda}
GPU_ID=${GPU_ID:-0}
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

MAIN_CKPT=${MAIN_CKPT:-"${MEDTRAJECTORY_EXTERNAL_ROOT}"/ckpt/MedTrajectory_exp2_tte_multitask/ckpt.pt}
MODERN_CKPT=${MODERN_CKPT:-"${MEDTRAJECTORY_EXTERNAL_ROOT}"/ckpt/MedTrajectory_exp2_modern_baseline/ckpt.pt}
NO_ROPE_CKPT=${NO_ROPE_CKPT:-"${MEDTRAJECTORY_EXTERNAL_ROOT}"/ckpt/MedTrajectory_exp2_modern_no_rope/ckpt.pt}
BLOCK48_CKPT=${BLOCK48_CKPT:-"${MEDTRAJECTORY_EXTERNAL_ROOT}"/ckpt/MedTrajectory_exp2_modern_block48/ckpt.pt}

TRAIN_ITERS=${TRAIN_ITERS:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-300}
EVAL_ITERS=${EVAL_ITERS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
BOOTSTRAP=${BOOTSTRAP:-500}

train_horizon() {
  local name="$1"
  local ckpt="$2"
  local block="$3"
  local next_w="$4"
  local tte_w="$5"
  local age_encoding="$6"
  local age_rope="$7"
  local out="results/high_priority_ablation/${name}"
  if [[ -f "$out/ckpt.pt" && -f "$out/history.json" ]]; then
    echo "[skip train] $name"
  else
    echo "[train horizon] $name"
    "$PY" src/semantic_delphi_ukb/train_medtrajectory_horizon_risk.py \
      --device "$DEVICE" \
      --init-from-ckpt "$ckpt" \
      --max-patients 0 \
      --max-iters "$TRAIN_ITERS" \
      --eval-interval "$EVAL_INTERVAL" \
      --eval-iters "$EVAL_ITERS" \
      --batch-size "$BATCH_SIZE" \
      --block-size "$block" \
      --learning-rate 3e-4 \
      --weight-decay 0.1 \
      --next-event-loss-weight "$next_w" \
      --tte-loss-weight "$tte_w" \
      --horizon-risk-loss-weight 1.0 \
      --use-age-encoding "$age_encoding" \
      --use-age-rope "$age_rope" \
      --horizons 5,10 \
      --seed 42 \
      --out-dir "$out" 2>&1 | tee "$out.log"
  fi
}

eval_horizon() {
  local name="$1"
  local block="$2"
  local out="results/locked_test_horizon_risk/high_priority_${name}"
  if [[ -f "$out/aggregate_metrics.csv" && -f "$out/raw_predictions.csv" ]]; then
    echo "[skip eval] $name"
  else
    echo "[eval horizon] $name"
    "$PY" src/semantic_delphi_ukb/evaluate_horizon_risk_locked_test.py \
      --model medtrajectory \
      --checkpoint "results/high_priority_ablation/${name}/ckpt.pt" \
      --split test \
      --eval-all \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --block-size "$block" \
      --bootstrap "$BOOTSTRAP" \
      --calibration-bins 10 \
      --out-dir "$out" 2>&1 | tee "$out.log"
  fi
}

train_survival() {
  local name="$1"
  local bins="$2"
  local bin_years="$3"
  local pos_w="$4"
  local focal="$5"
  local aux_w="$6"
  local next_w="$7"
  local out="results/high_priority_survival/${name}"
  if [[ -f "$out/ckpt.pt" && -f "$out/history.json" ]]; then
    echo "[skip survival train] $name"
  else
    echo "[train survival] $name"
    "$PY" src/semantic_delphi_ukb/train_medtrajectory_survival_horizon.py \
      --device "$DEVICE" \
      --init-from-ckpt "$MAIN_CKPT" \
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
      --survival-loss-weight 1.0 \
      --survival-pos-weight "$pos_w" \
      --survival-focal-gamma "$focal" \
      --aux-horizon-loss-weight "$aux_w" \
      --survival-bins "$bins" \
      --survival-bin-years "$bin_years" \
      --horizons 5,10 \
      --seed 42 \
      --out-dir "$out" 2>&1 | tee "$out.log"
  fi
}

eval_survival() {
  local name="$1"
  local out="results/locked_test_horizon_risk/high_priority_survival_${name}"
  if [[ -f "$out/aggregate_metrics.csv" && -f "$out/raw_predictions.csv" ]]; then
    echo "[skip survival eval] $name"
  else
    echo "[eval survival] $name"
    "$PY" src/semantic_delphi_ukb/evaluate_horizon_risk_locked_test.py \
      --model survival \
      --checkpoint "results/high_priority_survival/${name}/ckpt.pt" \
      --split test \
      --eval-all \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --block-size 128 \
      --bootstrap "$BOOTSTRAP" \
      --calibration-bins 10 \
      --out-dir "$out" 2>&1 | tee "$out.log"
  fi
}

mkdir -p results/high_priority_ablation results/high_priority_survival results/locked_test_horizon_risk

# Core complete ablations: same 3000-iter budget, same evaluator.
train_horizon "tte_trunk_aux02" "$MAIN_CKPT" 128 0.2 0.0 auto auto
eval_horizon "tte_trunk_aux02" 128

train_horizon "modern_trunk_full3000" "$MODERN_CKPT" 128 0.2 0.0 auto auto
eval_horizon "modern_trunk_full3000" 128

train_horizon "no_rope_trunk_full3000" "$NO_ROPE_CKPT" 128 0.2 0.0 auto auto
eval_horizon "no_rope_trunk_full3000" 128

train_horizon "block48_trunk_full3000" "$BLOCK48_CKPT" 48 0.2 0.0 auto auto
eval_horizon "block48_trunk_full3000" 48

# Time-encoding ablations from the same TTE checkpoint.
train_horizon "age_only_no_rope" "$MAIN_CKPT" 128 0.2 0.0 true false
eval_horizon "age_only_no_rope" 128

train_horizon "rope_only_no_age" "$MAIN_CKPT" 128 0.2 0.0 false true
eval_horizon "rope_only_no_age" 128

train_horizon "no_age_no_rope" "$MAIN_CKPT" 128 0.2 0.0 false false
eval_horizon "no_age_no_rope" 128

# Auxiliary objective ablations on the strongest trunk.
train_horizon "tte_trunk_no_next_aux" "$MAIN_CKPT" 128 0.0 0.0 auto auto
eval_horizon "tte_trunk_no_next_aux" 128

train_horizon "tte_trunk_next_aux05" "$MAIN_CKPT" 128 0.5 0.0 auto auto
eval_horizon "tte_trunk_next_aux05" 128

# Survival-horizon can become the main model only if a full run closes the locked-test gap.
train_survival "surv_bin1y_pos10_aux02_full3000" 10 1.0 10.0 1.0 0.2 0.2
eval_survival "surv_bin1y_pos10_aux02_full3000"

train_survival "surv_bin2y_pos10_aux02_full3000" 5 2.0 10.0 1.0 0.2 0.2
eval_survival "surv_bin2y_pos10_aux02_full3000"

echo "[done] high-priority ablation matrix"
