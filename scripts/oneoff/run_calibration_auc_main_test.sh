#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="${MEDTRAJECTORY_ROOT}"
python_bin=""${MEDTRAJECTORY_PYTHON:-python3}""
run_dir="${1:-${project_root}/results/calibration_auc/monotonic_gated_rope_nextw0p2_test_full_20260805}"
checkpoint="${project_root}/results/next_event_weight_sweep/monotonic_gated_rope_nextw0p2/ckpt.pt"

mkdir -p "${run_dir}"
cd "${project_root}"

command=(
  "${python_bin}" -u src/semantic_delphi_ukb/evaluate_calibration_auc.py
  --checkpoint "${checkpoint}"
  --split test
  --out-dir "${run_dir}"
  --device cuda
  --batch-size 128
  --max-patients 0
  --filter-min-total 100
  --disease-chunk-size 200
  --age-groups 40,45,50,55,60,65,70,75
  --offset 0.1
  --seed 1337
)

printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
printf '\n' >> "${run_dir}/command.txt"
started_at="$(date -Iseconds)"
printf '{"state":"running","started_at":"%s"}\n' "${started_at}" > "${run_dir}/status.json.tmp"
mv "${run_dir}/status.json.tmp" "${run_dir}/status.json"

"${command[@]}" > "${run_dir}/stdout.log" 2>&1
exit_code=$?
printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"
finished_at="$(date -Iseconds)"
if [[ ${exit_code} -eq 0 ]]; then
  state="finished"
else
  state="failed"
fi
printf '{"state":"%s","started_at":"%s","finished_at":"%s","exit_code":%d}\n' \
  "${state}" "${started_at}" "${finished_at}" "${exit_code}" > "${run_dir}/status.json.tmp"
mv "${run_dir}/status.json.tmp" "${run_dir}/status.json"
exit "${exit_code}"
