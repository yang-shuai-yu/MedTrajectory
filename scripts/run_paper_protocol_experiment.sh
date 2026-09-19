#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

experiment_key="${1:?usage: run_paper_protocol_experiment.sh P0|P1|P2|P3|P4 TRAINING_SEED [run_dir]}"
training_seed="${2:?training seed is required}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
paper_data_root="${PAPER_DATA_ROOT:-"${MEDTRAJECTORY_DATA_ROOT}"/paper_protocol_v1}"
config="${project_root}/configs/paper_protocol_v1/${experiment_key}.json"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
run_root="${3:-${project_root}/results/paper_protocol_v1/${experiment_key}_seed${training_seed}_${timestamp}}"
device="${PAPER_DEVICE:-auto}"
gpu_id="${PAPER_GPU_ID:-0}"

case "${training_seed}" in
  42|43|44) ;;
  1337) echo "warning: seed 1337 is retained only for legacy launcher compatibility" >&2 ;;
  *) echo "training seed must be one of 42, 43, 44" >&2; exit 2 ;;
esac
test -f "${config}"
test -x "${python_bin}"
profile="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["dataset_profile"])' "${config}")"
data_dir="${paper_data_root}/${profile}"
test -f "${data_dir}/prepare_manifest.json"

if [[ "${device}" == "auto" ]]; then
  device="cpu"
  if command -v nvidia-smi >/dev/null 2>&1; then
    used_memory="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used_memory}" =~ ^[0-9]+$ ]] && (( used_memory < 1024 )); then
      device="cuda"
    fi
  fi
fi
if [[ "${device}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

mkdir -p "${run_root}/pretraining" "${run_root}/risk" "${run_root}/medical_auc" "${run_root}/longitudinal_auc"
printf 'experiment=%s\ntraining_seed=%s\ndata_dir=%s\ndevice=%s\ngpu_id=%s\n' \
  "${experiment_key}" "${training_seed}" "${data_dir}" "${device}" "${gpu_id}" > "${run_root}/resource_decision.txt"
cd "${project_root}"

stage_finished() {
  local stage_dir="$1"
  [[ -f "${stage_dir}/status.json" ]] && "${python_bin}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "finished" else 1)' \
    "${stage_dir}/status.json"
}

if ! stage_finished "${run_root}/pretraining"; then
  pretrain_cmd=("${python_bin}" -u src/semantic_delphi_ukb/train_paper_protocol_pretraining.py
    --experiment-config "${config}" --data-dir "${data_dir}" --seed "${training_seed}"
    --run-dir "${run_root}/pretraining" --device "${device}")
  if [[ -f "${run_root}/pretraining/checkpoints/last.pt" ]]; then
    pretrain_cmd+=(--resume "${run_root}/pretraining/checkpoints/last.pt")
  fi
  "${pretrain_cmd[@]}" 2>&1 | tee -a "${run_root}/pretraining/stdout.log"
fi

pretrain_ckpt="${run_root}/pretraining/checkpoints/best_val_loss.pt"
test -f "${pretrain_ckpt}"
if ! stage_finished "${run_root}/risk"; then
  risk_cmd=("${python_bin}" -u src/semantic_delphi_ukb/train_paper_protocol_risk_heads.py
    --experiment-config "${config}" --data-dir "${data_dir}" --seed "${training_seed}"
    --init-from-ckpt "${pretrain_ckpt}" --run-dir "${run_root}/risk" --device "${device}")
  if [[ -f "${run_root}/risk/checkpoints/last.pt" ]]; then
    risk_cmd+=(--resume "${run_root}/risk/checkpoints/last.pt")
  fi
  "${risk_cmd[@]}" 2>&1 | tee -a "${run_root}/risk/stdout.log"
fi

risk_ckpt="${run_root}/risk/checkpoints/best_val_loss.pt"
test -f "${risk_ckpt}"
if [[ ! -f "${run_root}/medical_auc/summary.json" ]]; then
  "${python_bin}" -u src/semantic_delphi_ukb/evaluate_paper_protocol_medical_auc.py \
    --checkpoint "${risk_ckpt}" --data-dir "${data_dir}" --device "${device}" \
    --out-dir "${run_root}/medical_auc" 2>&1 | tee -a "${run_root}/medical_auc/stdout.log"
fi

if [[ ! -f "${run_root}/longitudinal_auc/summary.json" ]]; then
  "${python_bin}" -u src/semantic_delphi_ukb/evaluate_paper_protocol_longitudinal.py \
    --checkpoint "${risk_ckpt}" --data-dir "${data_dir}" --device "${device}" \
    --out-dir "${run_root}/longitudinal_auc" 2>&1 | tee -a "${run_root}/longitudinal_auc/stdout.log"
fi

if [[ "${experiment_key}" == "P0" || "${experiment_key}" == "P3" ]]; then
  mkdir -p "${run_root}/survival"
  if ! stage_finished "${run_root}/survival"; then
    survival_cmd=("${python_bin}" -u src/semantic_delphi_ukb/train_paper_protocol_survival_horizon.py
      --experiment-config "${config}" --data-dir "${data_dir}" --seed "${training_seed}"
      --init-from-ckpt "${pretrain_ckpt}" --run-dir "${run_root}/survival" --device "${device}")
    if [[ -f "${run_root}/survival/checkpoints/last.pt" ]]; then
      survival_cmd+=(--resume "${run_root}/survival/checkpoints/last.pt")
    fi
    "${survival_cmd[@]}" 2>&1 | tee -a "${run_root}/survival/stdout.log"
  fi
fi
