#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

session="${1:?session name is required}"
freeze_manifest="${2:?freeze manifest is required}"
split_audit="${3:?split audit is required}"
out_root="${4:?output root is required}"
spec_file="${5:?model spec TSV is required}"
consistency_list="${6:?validation consistency list is required}"
device="${7:-cuda}"
landmark_manifest="${8:-${PAPER_LANDMARK_MANIFEST:-}}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
diseases_yaml="${PAPER_DISEASES_YAML:-${project_root}/docs/selected_diseases.yaml}"

test -f "${freeze_manifest}"
test -f "${split_audit}"
test -f "${spec_file}"
test -f "${consistency_list}"
if [[ -n "${landmark_manifest}" ]]; then test -f "${landmark_manifest}"; fi
if [[ -e "${out_root}" ]]; then
  echo "locked-test output root already exists: ${out_root}" >&2
  exit 4
fi
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 5
fi
mkdir -p "${out_root}"

run_cmd="set -euo pipefail
cd '${project_root}'
'${python_bin}' -u scripts/verify_paper_protocol_freeze_manifest.py --freeze-manifest '${freeze_manifest}' --out '${out_root}/freeze_verification.json'
while IFS=\$'\t' read -r name mode checkpoint data_dir probe; do
  [[ -z \"\${name}\" || \"\${name:0:1}\" == \"#\" ]] && continue
  out_dir='${out_root}/'\"\${name}\"
  mkdir -p \"\${out_dir}\"
  eval_args=(--mode \"\${mode}\" --checkpoint \"\${checkpoint}\" --data-dir \"\${data_dir}\" --split test --diseases-yaml '${diseases_yaml}' --out-dir \"\${out_dir}\" --device '${device}' --horizons 1,5,10 --age-groups 50,55,60,65,70,75 --seed 1337)
  if [[ -n '${landmark_manifest}' ]]; then eval_args+=(--landmark-manifest '${landmark_manifest}'); fi
  if [[ -n \"\${probe:-}\" ]]; then eval_args+=(--probe-checkpoint \"\${probe}\"); fi
  '${python_bin}' -u src/semantic_delphi_ukb/evaluate_medical_control_tasks.py \"\${eval_args[@]}\" 2>&1 | tee \"\${out_dir}/stdout.log\"
done < '${spec_file}'

gate_args=(--freeze-manifest '${freeze_manifest}' --freeze-verification '${out_root}/freeze_verification.json' --split-audit '${split_audit}' --out-dir '${out_root}' --bootstrap 1000 --seed 42 --memory-efficient)
if [[ -n '${landmark_manifest}' ]]; then gate_args+=(--landmark-manifest '${landmark_manifest}'); fi
while IFS= read -r consistency; do
  [[ -z \"\${consistency}\" || \"\${consistency:0:1}\" == \"#\" ]] && continue
  gate_args+=(--consistency \"\${consistency}\" --require-consistency \"\$(basename \"\${consistency}\" .json)\")
done < '${consistency_list}'
while IFS=\$'\t' read -r name mode checkpoint data_dir probe; do
  [[ -z \"\${name}\" || \"\${name:0:1}\" == \"#\" ]] && continue
  gate_args+=(--input \"\${name}=${out_root}/\${name}/rows.json\")
done < '${spec_file}'
'${python_bin}' -u scripts/run_locked_test_paper_protocol.py \"\${gate_args[@]}\"
"
tmux new-session -d -s "${session}" "${run_cmd} > '${out_root}/stdout.log' 2>&1"
echo "session=${session}"
echo "out_root=${out_root}"
