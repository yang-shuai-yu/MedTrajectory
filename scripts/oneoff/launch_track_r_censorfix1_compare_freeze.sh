#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

root="${MEDTRAJECTORY_ROOT}"
python="${MEDTRAJECTORY_PYTHON:-python3}"
run="$root/results/track_r_v2_1/runs/seed42_censorfix1"
data="${MEDTRAJECTORY_DATA_ROOT}"/data/track_r_v2_1/multitype_static_prefix

models=(
  A0-TokenStatic
  A0-noStatic
  A1-TokenStatic
  A1-noStatic
  Logistic-R
  Cox-R
  MDRMF-Clinical-R
  Med-BERT-Paper
  Med-BERT-Matched-S
)

cd "$root"
export PYTHONPATH=src

for model in "${models[@]}"; do
  summary="$run/$model/validation/summary.json"
  rows="$run/$model/validation/rows.json.gz"
  test -s "$summary"
  test -s "$rows"
  gzip -t "$rows"
done

compare_args=()
for model in "${models[@]}"; do
  compare_args+=(--input "$model=$run/$model/validation/rows.json.gz")
done

"$python" -u -m semantic_delphi_ukb.compare_horizon_control_tasks \
  "${compare_args[@]}" \
  --out-dir "$run/comparison" \
  --bootstrap 1000 \
  --seed 42 \
  --calibration-bins 10 \
  --memory-efficient \
  --factorial-interaction A1-TokenStatic,A0-TokenStatic,A1-noStatic,A0-noStatic

"$python" -u scripts/freeze_track_r_validation.py \
  --protocol "$root/configs/paper_protocol_v1/TRACK_R_v2_1.json" \
  --data-manifest "$data/prepare_manifest.json" \
  --field-audit "$root/results/track_r_v2_1/manifests/static_field_audit_passed_20260813.json" \
  --capacity-report "$root/results/track_r_v2_1/manifests/model_capacity.json" \
  --validation-landmarks "$root/results/track_r_v2_1/manifests/val_shared_landmarks.json" \
  --model "A0-TokenStatic=$root/results/track_r_v2_1/runs/seed42_retry1/A0-TokenStatic/horizon/checkpoints/best_val_horizon_auc.pt=$run/A0-TokenStatic/validation/rows.json.gz" \
  --model "A0-noStatic=$root/results/track_r_v2_1/diagnostics/carope_static_integration/seed42_diagnostic2/A0-noStatic/horizon/checkpoints/best_val_horizon_auc.pt=$run/A0-noStatic/validation/rows.json.gz" \
  --model "A1-TokenStatic=$root/results/track_r_v2_1/runs/seed42_retry1/A1-TokenStatic/horizon/checkpoints/best_val_horizon_auc.pt=$run/A1-TokenStatic/validation/rows.json.gz" \
  --model "A1-noStatic=$root/results/track_r_v2_1/runs/seed42_retry1/A1-noStatic/horizon/checkpoints/best_val_horizon_auc.pt=$run/A1-noStatic/validation/rows.json.gz" \
  --model "Logistic-R=$root/results/track_r_v2_1/runs/seed42/Logistic-R/validation/checkpoint.pkl=$run/Logistic-R/validation/rows.json.gz" \
  --model "Cox-R=$run/Cox-R/validation/checkpoint.pkl=$run/Cox-R/validation/rows.json.gz" \
  --model "MDRMF-Clinical-R=$run/MDRMF-Clinical-R/validation/checkpoint.pt=$run/MDRMF-Clinical-R/validation/rows.json.gz" \
  --model "Med-BERT-Paper=$root/results/track_r_v2_1/runs/seed42_retry1/Med-BERT-Paper/horizon/checkpoints/best_val_horizon_auc.pt=$run/Med-BERT-Paper/validation/rows.json.gz" \
  --model "Med-BERT-Matched-S=$root/results/track_r_v2_1/runs/seed42_retry1/Med-BERT-Matched-S/horizon/checkpoints/best_val_horizon_auc.pt=$run/Med-BERT-Matched-S/validation/rows.json.gz" \
  --output "$run/validation_freeze_manifest.json"

echo '{"status":"finished","comparison":"comparison/comparison.json","freeze":"validation_freeze_manifest.json"}'
