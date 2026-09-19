#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="${MEDTRAJECTORY_ROOT}"
python_bin=""${MEDTRAJECTORY_PYTHON:-python3}""

session="${1:?session name required}"
variant="${2:?variant name required}"
use_age_encoding="${3:?use_age_encoding required}"
use_age_rope="${4:?use_age_rope required}"
use_relative_query="${5:?use_relative_horizon_query required}"
gap_weight="${6:?time_gap_loss_weight required}"

pretrain_dir="${project_root}/results/paper_protocol_v1/car_rope_validation_20260810/${variant}/pretraining"
run_dir="${project_root}/results/paper_protocol_v1/car_rope_validation_20260810/${variant}/horizon"
init_ckpt="${pretrain_dir}/checkpoints/best_val_loss.pt"
test -f "${init_ckpt}" || { echo "missing pretraining checkpoint: ${init_ckpt}" >&2; exit 3; }
mkdir -p "${run_dir}"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "session already exists: ${session}" >&2
  exit 2
fi

command="cd '${project_root}' && export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 && '${python_bin}' -u src/semantic_delphi_ukb/train_car_rope.py --data-dir '${project_root}/data/paper_protocol_v1/multitype' --diseases-yaml '${project_root}/docs/selected_diseases.yaml' --run-dir '${run_dir}' --init-from-ckpt '${init_ckpt}' --device cuda --seed 42 --max-iters 10000 --eval-interval 250 --use-age-encoding '${use_age_encoding}' --use-age-rope '${use_age_rope}' --use-relative-horizon-query '${use_relative_query}' --time-gap-loss-weight '${gap_weight}' > '${run_dir}/stdout.log' 2>&1"
tmux new-session -d -s "${session}" "${command}"

echo "started=${session}"
echo "run_dir=${run_dir}"
