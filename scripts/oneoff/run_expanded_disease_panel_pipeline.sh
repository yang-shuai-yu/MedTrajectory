#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ROOT="${MEDTRAJECTORY_ROOT}"
PY=""${MEDTRAJECTORY_PYTHON:-python3}""
MAMBA_PY=""${MEDTRAJECTORY_PYTHON:-python3}""
DATA_DIR="$ROOT/data/ukb_semantic_multitype_explicit_split"

cd "$ROOT"

"$PY" src/semantic_delphi_ukb/summarize_expanded_disease_panel.py \
  --data-dir "$DATA_DIR" \
  --diseases-yaml docs/expanded_disease_panel_ukb_icd10.yaml \
  --out-dir results/expanded_disease_panel \
  --selection right --padding regular --block-size 128 --batch-size 256 --device cpu

"$PY" src/semantic_delphi_ukb/evaluate_expanded_disease_topk.py \
  --data-dir "$DATA_DIR" \
  --diseases-yaml docs/expanded_disease_panel_ukb_icd10.yaml \
  --counts-wide results/expanded_disease_panel/expanded_disease_panel_counts_wide.csv \
  --out-dir results/expanded_disease_panel_topk \
  --prefix expanded_disease_topk_nomamba \
  --models monotonic,gated,no_rope,old_tte,bert \
  --selection right --padding regular --block-size 128 --batch-size 256 --device cpu

CUDA_VISIBLE_DEVICES=0 "$MAMBA_PY" src/semantic_delphi_ukb/evaluate_expanded_disease_topk.py \
  --data-dir "$DATA_DIR" \
  --diseases-yaml docs/expanded_disease_panel_ukb_icd10.yaml \
  --counts-wide results/expanded_disease_panel/expanded_disease_panel_counts_wide.csv \
  --out-dir results/expanded_disease_panel_topk \
  --prefix expanded_disease_topk_mamba \
  --models mamba \
  --selection right --padding regular --block-size 128 --batch-size 256 --device cuda

CUDA_VISIBLE_DEVICES=0 "$PY" src/semantic_delphi_ukb/train_medtrajectory_horizon_risk.py \
  --device cuda \
  --data-dir "$DATA_DIR" \
  --diseases-yaml docs/expanded_disease_panel_ukb_icd10.yaml \
  --init-from-ckpt results/monotonic_horizon/monotonic_gated_rope_gate100/ckpt.pt \
  --out-dir results/expanded_horizon_risk/medtrajectory_monotonic_38d_full3000 \
  --max-patients 0 --max-iters 3000 --eval-interval 300 --eval-iters 100 \
  --batch-size 128 --block-size 128 --learning-rate 3e-4 --weight-decay 0.1 \
  --next-event-loss-weight 0.2 --horizon-risk-loss-weight 1.0 --horizons 5,10 \
  --use-age-encoding true --use-age-rope true --age-rope-gate 1.0 --monotonic-horizon-risk --seed 42

CUDA_VISIBLE_DEVICES=0 "$PY" src/semantic_delphi_ukb/train_architecture_risk_heads.py \
  --model bert --device cuda \
  --data-dir "$DATA_DIR" \
  --diseases-yaml docs/expanded_disease_panel_ukb_icd10.yaml \
  --out-dir results/expanded_horizon_risk/bert_38d_full3000 \
  --max-patients 0 --max-iters 3000 --eval-interval 300 --eval-iters 100 \
  --batch-size 128 --block-size 128 --n-layer 4 --n-head 8 --n-embd 64 \
  --learning-rate 3e-4 --weight-decay 0.1 --horizons 5,10 --seed 42
