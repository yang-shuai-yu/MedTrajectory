#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${MEDTRAJECTORY_ROOT}}"
PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/data/ukb_semantic_multitype_explicit_split}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT"

"$PY" scripts/build_icd10_hierarchy_panel.py \
  --data-dir "$DATA_DIR" \
  --out docs/icd10_hierarchy_panel.yaml \
  --summary-csv results/icd10_hierarchy_panel/icd10_hierarchy_panel_summary.csv \
  --min-tokens 1

"$PY" src/semantic_delphi_ukb/summarize_icd10_hierarchy_panel.py \
  --data-dir "$DATA_DIR" \
  --panel docs/icd10_hierarchy_panel.yaml \
  --out results/icd10_hierarchy_panel/icd10_hierarchy_panel_token_summary.csv

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PY" src/semantic_delphi_ukb/train_medtrajectory_horizon_risk.py \
  --device "$DEVICE" \
  --data-dir "$DATA_DIR" \
  --diseases-yaml docs/icd10_hierarchy_panel.yaml \
  --init-from-ckpt results/monotonic_horizon/monotonic_gated_rope_gate100/ckpt.pt \
  --out-dir results/icd10_hierarchy_panel/monotonic_gated_rope_hierarchy_full3000 \
  --max-patients 0 --max-iters 3000 --eval-interval 300 --eval-iters 100 \
  --batch-size 128 --block-size 128 --learning-rate 3e-4 --weight-decay 0.1 \
  --next-event-loss-weight 0.2 --horizon-risk-loss-weight 1.0 --horizons 5,10 \
  --use-age-encoding true --use-age-rope true --age-rope-gate 1.0 --monotonic-horizon-risk --seed 42

