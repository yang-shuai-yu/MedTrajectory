#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

REPO="${MEDTRAJECTORY_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
MATRIX=$REPO/results/paper_protocol_v1/matrix_seeds42-44_my3_20260806_0252
OUT=$REPO/results/paper_protocol_v1/horizon_controls_validation_20260808
SRC=$REPO/src
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-20000}
COMMON=(env PYTHONPATH="$SRC" "$PY")

mkdir -p "$OUT"

write_status() {
  local path=$1
  local state=$2
  local stage=${3:-}
  local tmp="${path}.tmp"
  printf '{"state":"%s","stage":"%s","updated_utc":"%s"}\n' "$state" "$stage" "$(date -u +%FT%TZ)" > "$tmp"
  mv -f "$tmp" "$path"
}

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [ "${used:-999999}" -lt "$GPU_MEMORY_LIMIT_MIB" ]; then
      return
    fi
    sleep 30
  done
}

run_stage() {
  local run_dir=$1
  local stage=$2
  local marker=$3
  shift 3
  mkdir -p "$run_dir"
  if [ -f "$marker" ]; then
    return
  fi
  write_status "$run_dir/status.json" running "$stage"
  printf '%q ' "$@" > "$run_dir/${stage}.command"
  wait_for_gpu
  if "$@" > "$run_dir/${stage}.stdout.log" 2>&1; then
    touch "$marker"
  else
    write_status "$run_dir/status.json" failed "$stage"
    return 1
  fi
}

run_one() {
  local experiment=$1
  local seed=$2
  local profile=$3
  local run_dir="$OUT/${experiment}_seed${seed}"
  local data_dir="${MEDTRAJECTORY_DATA_ROOT}"/paper_protocol_v1/${profile}
  local base="$MATRIX/${experiment}_seed${seed}"
  run_stage "$run_dir" lm "$run_dir/lm/summary.json" \
    "${COMMON[@]}" "$SRC/semantic_delphi_ukb/evaluate_horizon_control_tasks.py" \
    --mode lm --checkpoint "$base/pretraining/checkpoints/best_val_loss.pt" \
    --data-dir "$data_dir" --split val --out-dir "$run_dir/lm" --device cuda \
    --batch-size 256 --block-size 128 --horizons 1,5,10

  run_stage "$run_dir" linear_train "$run_dir/linear_probe.pt" \
    "${COMMON[@]}" "$SRC/semantic_delphi_ukb/train_frozen_trunk_linear_probe.py" \
    --base-checkpoint "$base/pretraining/checkpoints/best_val_loss.pt" \
    --data-dir "$data_dir" --out-checkpoint "$run_dir/linear_probe.pt" \
    --device cuda --batch-size 256 --block-size 128 --horizons 1,5,10 \
    --max-iters 1000 --eval-interval 500 --seed "$seed"

  run_stage "$run_dir" linear "$run_dir/linear/summary.json" \
    "${COMMON[@]}" "$SRC/semantic_delphi_ukb/evaluate_horizon_control_tasks.py" \
    --mode linear --checkpoint "$base/pretraining/checkpoints/best_val_loss.pt" \
    --probe-checkpoint "$run_dir/linear_probe.pt" --data-dir "$data_dir" \
    --split val --out-dir "$run_dir/linear" --device cuda \
    --batch-size 256 --block-size 128 --horizons 1,5,10

  run_stage "$run_dir" explicit "$run_dir/explicit/summary.json" \
    "${COMMON[@]}" "$SRC/semantic_delphi_ukb/evaluate_horizon_control_tasks.py" \
    --mode explicit --checkpoint "$base/risk/checkpoints/best_val_loss.pt" \
    --data-dir "$data_dir" --split val --out-dir "$run_dir/explicit" --device cuda \
    --batch-size 256 --block-size 128 --horizons 1,5,10

  if [ "$experiment" = P0 ] || [ "$experiment" = P3 ]; then
    run_stage "$run_dir" survival "$run_dir/survival/summary.json" \
      "${COMMON[@]}" "$SRC/semantic_delphi_ukb/evaluate_horizon_control_tasks.py" \
      --mode survival --checkpoint "$base/survival/checkpoints/best_val_loss.pt" \
      --data-dir "$data_dir" --split val --out-dir "$run_dir/survival" --device cuda \
      --batch-size 256 --block-size 128 --horizons 1,5,10
  fi

  write_status "$run_dir/status.json" finished all_stages
}

write_status "$OUT/status.json" running validation_matrix

run_one P0 42 diagnosis_death
run_one P0 43 diagnosis_death
run_one P0 44 diagnosis_death
run_one P2 42 multitype
run_one P2 43 multitype
run_one P2 44 multitype
run_one P3 42 multitype
run_one P3 43 multitype
run_one P3 44 multitype

for experiment in P0 P2 P3; do
  for seed in 42 43 44; do
    run_dir="$OUT/${experiment}_seed${seed}"
    inputs=(--input "lm=$run_dir/lm/rows.json" --input "linear=$run_dir/linear/rows.json" --input "explicit=$run_dir/explicit/rows.json")
    if [ "$experiment" = P0 ] || [ "$experiment" = P3 ]; then
      inputs+=(--input "survival=$run_dir/survival/rows.json")
    fi
    run_stage "$run_dir" compare "$run_dir/compare/comparison.json" \
      "${COMMON[@]}" "$SRC/semantic_delphi_ukb/compare_horizon_control_tasks.py" \
      "${inputs[@]}" --out-dir "$run_dir/compare" --bootstrap 500 --seed "$seed"
  done
done

write_status "$OUT/status.json" finished all_runs
