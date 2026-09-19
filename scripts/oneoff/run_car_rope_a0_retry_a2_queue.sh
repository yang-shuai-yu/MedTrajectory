#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
data_dir="${MEDTRAJECTORY_DATA_ROOT}"/data/paper_protocol_v2_locked_test/multitype
run_root="${project_root}/results/paper_protocol_v2/car_rope_locked_retrain_20260811/seed42"
log_root="${project_root}/results/paper_protocol_v2/launcher_logs"
queue_log="${log_root}/carope_v2_my1_a0_retry_a2_20260811.log"
a0_log="${log_root}/carope_v2_a0_horizon_retry1_20260811.log"
a2_log="${log_root}/carope_v2_a2_all_20260811.log"

for path in "${queue_log}" "${a0_log}" "${a2_log}" \
  "${run_root}/A0_sincos_control/horizon_retry1" \
  "${run_root}/A2_relative_horizon_query"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing path: ${path}" >&2
    exit 2
  fi
done

exec >"${queue_log}" 2>&1
echo "starting_A0_horizon_retry1"
"${python_bin}" -u "${project_root}/src/semantic_delphi_ukb/train_car_rope.py" \
  --data-dir "${data_dir}" \
  --diseases-yaml "${project_root}/docs/selected_diseases.yaml" \
  --run-dir "${run_root}/A0_sincos_control/horizon_retry1" \
  --init-from-ckpt "${run_root}/A0_sincos_control/pretraining/checkpoints/best_val_loss.pt" \
  --device cuda --seed 42 --horizons 1.0,5.0,10.0 \
  --max-iters 10000 --eval-interval 250 \
  --use-age-encoding true --use-age-rope false \
  --use-relative-horizon-query false --time-gap-loss-weight 0.0 \
  >"${a0_log}" 2>&1

grep -q '"status": "finished"' "${run_root}/A0_sincos_control/horizon_retry1/status.json"
echo "starting_A2_all"
"${python_bin}" -u "${project_root}/scripts/run_car_rope_locked_retrain.py" \
  --spec "${project_root}/configs/paper_protocol_v1/CARoPE_locked_retrain_v2_my1.json" \
  --variants A2_relative_horizon_query --stage all --device cuda \
  --validate-inputs --allow-existing-output-root --execute \
  >"${a2_log}" 2>&1

echo "queue_finished"
