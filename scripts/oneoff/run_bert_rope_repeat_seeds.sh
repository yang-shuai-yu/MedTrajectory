#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "${MEDTRAJECTORY_ROOT}"
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:${PYTHONPATH:-}"

PY="${MEDTRAJECTORY_PYTHON:-python3}"

for seed in 43 44; do
  echo "[$(date)] training bert_rope seed ${seed}"
  CUDA_VISIBLE_DEVICES=0 "$PY" src/semantic_delphi_ukb/train_architecture_risk_heads.py \
    --model bert_rope --device cuda \
    --data-dir data/ukb_semantic_multitype_explicit_split \
    --out-dir "results/architecture_risk_heads/bert_rope_full_seed${seed}" \
    --max-patients 0 --max-iters 3000 --eval-interval 300 --eval-iters 100 \
    --batch-size 128 --block-size 128 --n-layer 4 --n-head 8 --n-embd 64 \
    --learning-rate 3e-4 --weight-decay 0.1 --horizons 5,10 --seed "$seed"

  echo "[$(date)] evaluating bert_rope seed ${seed}"
  CUDA_VISIBLE_DEVICES=0 "$PY" src/semantic_delphi_ukb/evaluate_horizon_risk_locked_test.py \
    --model bert_rope \
    --checkpoint "results/architecture_risk_heads/bert_rope_full_seed${seed}/ckpt.pt" \
    --split test --eval-all --device cuda --batch-size 128 --block-size 128 \
    --bootstrap 500 --calibration-bins 10 \
    --out-dir "results/locked_test_horizon_risk/bert_rope_test_seed${seed}"
done

echo "[$(date)] done"
