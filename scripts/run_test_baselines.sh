#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ROOT="${MEDTRAJECTORY_ROOT}"
cd "$ROOT"
export PYTHONPATH=src
PY=""${MEDTRAJECTORY_PYTHON:-python3}""
DATA="data/track_r_v2_1/multitype_static_prefix"
LM="results/track_r_v2_1/manifests/test_shared_landmarks.json"
FEAT="results/track_r_v2_1/landmark_features"
OUT="results/track_r_v2_2/test"

# statistical/clinical baselines: frozen checkpoint -> predict on test
"$PY" -m semantic_delphi_ukb.track_r_baselines --model cox \
  --train-features "$FEAT/train.npz" --val-features "$FEAT/val.npz" --eval-features "$FEAT/test.npz" \
  --checkpoint results/track_r_v2_1/runs/seed42_censorfix1/Cox-R/validation/checkpoint.pkl \
  --predict-only --output-dir "$OUT/baselines/cox" --device cpu

"$PY" -m semantic_delphi_ukb.track_r_baselines --model logistic \
  --train-features "$FEAT/train.npz" --val-features "$FEAT/val.npz" --eval-features "$FEAT/test.npz" \
  --checkpoint results/track_r_v2_1/runs/seed42/Logistic-R/validation/checkpoint.pkl \
  --predict-only --output-dir "$OUT/baselines/logistic" --device cpu

"$PY" -m semantic_delphi_ukb.track_r_baselines --model mdrmf-clinical \
  --train-features "$FEAT/train.npz" --val-features "$FEAT/val.npz" --eval-features "$FEAT/test.npz" \
  --checkpoint results/track_r_v2_1/runs/seed42_censorfix1/MDRMF-Clinical-R/validation/checkpoint.pt \
  --predict-only --output-dir "$OUT/baselines/mdrmf" --device cpu

# Med-BERT baselines: forward on test
"$PY" -m semantic_delphi_ukb.evaluate_track_r --family medbert --model-name Med-BERT-Paper \
  --checkpoint results/track_r_v2_1/runs/seed42_retry1/Med-BERT-Paper/horizon/checkpoints/best_val_horizon_auc.pt \
  --protocol configs/paper_protocol_v1/TRACK_R_v2_1.json --data-dir "$DATA" --split test \
  --landmark-manifest "$LM" --out-dir "$OUT/baselines/medbert_paper" \
  --include-static-prefix true --device cuda

"$PY" -m semantic_delphi_ukb.evaluate_track_r --family medbert --model-name Med-BERT-Matched-S \
  --checkpoint results/track_r_v2_1/runs/seed42_retry1/Med-BERT-Matched-S/horizon/checkpoints/best_val_horizon_auc.pt \
  --protocol configs/paper_protocol_v1/TRACK_R_v2_1.json --data-dir "$DATA" --split test \
  --landmark-manifest "$LM" --out-dir "$OUT/baselines/medbert_matched" \
  --include-static-prefix true --device cuda

echo "ALL_DONE"
