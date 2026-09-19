#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
data_dir="${MEDTRAJECTORY_DATA_ROOT}"/data/paper_protocol_v2_locked_test/multitype
train_root="${project_root}/results/paper_protocol_v2/car_rope_locked_retrain_20260811/seed42"
validation_run_id="${CAROPE_VALIDATION_RUN_ID:-car_rope_locked_validation_20260811}"
root="${project_root}/results/paper_protocol_v2/${validation_run_id}"
audit="${project_root}/results/paper_protocol_v2/split_audits/multitype.json"
protocol="${project_root}/configs/paper_protocol_v1/CARoPE_locked_test_protocol_v2.json"
test_eids="${project_root}/data/paper_protocol_v1/locked_test_eids.csv"
bundle="${root}/validation_official_aggregates.json"
consistency="${root}/validation_consistency.json"
effect_gate="${root}/validation_effect_gate.json"
freeze="${root}/freeze_manifest.json"
verification="${root}/freeze_verification.json"
status="${root}/freeze_status.json"
log="${root}/freeze.stdout.log"

for path in "${bundle}" "${consistency}" "${effect_gate}" "${freeze}" "${verification}" "${status}" "${log}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing path: ${path}" >&2
    exit 2
  fi
done

write_status() {
  local state="$1"
  local exit_code="${2:-null}"
  local temp="${status}.tmp"
  printf '{"state":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${state}" "${exit_code}" "$(date -u +%FT%TZ)" > "${temp}"
  mv -f "${temp}" "${status}"
}

run_pipeline() {
  "${python_bin}" "${project_root}/scripts/check_car_rope_validation_gate.py" \
    --comparison "${root}/validation_comparison/comparison.json" --out "${effect_gate}" || return $?
  "${python_bin}" "${project_root}/scripts/build_validation_official_aggregate_bundle.py" \
    --input "A0=${root}/validation_official/A0_sincos_control/calibration_auc_aggregates.json" \
    --input "A1=${root}/validation_official/A1_car_rope_trunk/calibration_auc_aggregates.json" \
    --input "A2=${root}/validation_official/A2_relative_horizon_query/calibration_auc_aggregates.json" \
    --input "A3=${root}/validation_official/A3_full_carope/calibration_auc_aggregates.json" \
    --out "${bundle}" || return $?
  "${python_bin}" "${project_root}/scripts/check_paper_medical_control_consistency.py" \
    --control "A0=${root}/A0_sincos_control/summary.json" --require A0 \
    --control "A1=${root}/A1_car_rope_trunk/summary.json" --require A1 \
    --control "A2=${root}/A2_relative_horizon_query/summary.json" --require A2 \
    --control "A3=${root}/A3_full_carope/summary.json" --require A3 \
    --official-aggregates "${bundle}" --protocol-json "${protocol}" --out "${consistency}" || return $?
  "${python_bin}" "${project_root}/scripts/build_paper_protocol_freeze_manifest.py" \
    --split-audit "${audit}" --data-dir "${data_dir}" \
    --diseases-yaml "${project_root}/docs/selected_diseases.yaml" \
    --checkpoint "A0=${train_root}/A0_sincos_control/horizon_retry1/checkpoints/best_val_horizon_auc.pt" \
    --checkpoint "A1=${train_root}/A1_car_rope_trunk/horizon/checkpoints/best_val_horizon_auc.pt" \
    --checkpoint "A2=${train_root}/A2_relative_horizon_query/horizon/checkpoints/best_val_horizon_auc.pt" \
    --checkpoint "A3=${train_root}/A3_full_carope/horizon/checkpoints/best_val_horizon_auc.pt" \
    --asset "medical_control_evaluator=${project_root}/src/semantic_delphi_ukb/evaluate_medical_control_tasks.py" \
    --asset "comparison_evaluator=${project_root}/src/semantic_delphi_ukb/compare_horizon_control_tasks.py" \
    --asset "validation_consistency=${consistency}" \
    --asset "validation_official_aggregates=${bundle}" \
    --asset "validation_effect_gate=${effect_gate}" \
    --asset "vocab=${data_dir}/vocab/dynamic_token_vocab.csv" \
    --asset "shared_validation_landmarks=${root}/val_shared_landmarks.json" \
    --asset "shared_test_landmarks=${root}/test_shared_landmarks.json" \
    --model-data-dir "A0=${data_dir}" --model-split-audit "A0=${audit}" \
    --model-data-dir "A1=${data_dir}" --model-split-audit "A1=${audit}" \
    --model-data-dir "A2=${data_dir}" --model-split-audit "A2=${audit}" \
    --model-data-dir "A3=${data_dir}" --model-split-audit "A3=${audit}" \
    --protocol-json "${protocol}" --test-eids-csv "${test_eids}" \
    --confirm-checkpoint-excludes-test --out "${freeze}" || return $?
  "${python_bin}" "${project_root}/scripts/verify_paper_protocol_freeze_manifest.py" \
    --freeze-manifest "${freeze}" --out "${verification}" || return $?
}

write_status running
if PYTHONPATH="${project_root}/src" run_pipeline > "${log}" 2>&1; then
  write_status finished 0
else
  code=$?
  write_status failed "${code}"
  exit "${code}"
fi
