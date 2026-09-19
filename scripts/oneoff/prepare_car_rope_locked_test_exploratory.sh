#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PAPER_PYTHON_BIN:-"${MEDTRAJECTORY_PYTHON:-python3}"}"
run_id="${1:?locked-test run id is required}"
validation_run_id="${CAROPE_VALIDATION_RUN_ID:-car_rope_locked_validation_20260811_r1}"
data_dir="${MEDTRAJECTORY_DATA_ROOT}"/data/paper_protocol_v2_locked_test/multitype
train_root="${project_root}/results/paper_protocol_v2/car_rope_locked_retrain_20260811/seed42"
validation_root="${project_root}/results/paper_protocol_v2/${validation_run_id}"
root="${project_root}/results/paper_protocol_v2/${run_id}"
audit="${project_root}/results/paper_protocol_v2/split_audits/multitype.json"
protocol="${project_root}/configs/paper_protocol_v1/CARoPE_locked_test_protocol_v2.json"
test_eids="${project_root}/data/paper_protocol_v1/locked_test_eids.csv"
effect_gate="${validation_root}/validation_effect_gate.json"
authorization="${root}/exploratory_gate_override.json"
landmarks="${root}/test_shared_landmarks.json"
freeze="${root}/freeze_manifest.json"
verification="${root}/freeze_verification.json"
status="${root}/prepare_status.json"
log="${root}/prepare.stdout.log"

if [[ -e "${root}" ]]; then
  echo "refusing to overwrite locked-test root: ${root}" >&2
  exit 2
fi
mkdir -p "${root}"

write_status() {
  local state="$1"
  local exit_code="${2:-null}"
  local temp="${status}.tmp"
  printf '{"state":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${state}" "${exit_code}" "$(date -u +%FT%TZ)" > "${temp}"
  mv -f "${temp}" "${status}"
}

run_prepare() {
  "${python_bin}" - "${effect_gate}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("ok") is not False:
    raise SystemExit("exploratory override requires a recorded failed validation gate")
PY

  "${python_bin}" - "${authorization}" "${effect_gate}" <<'PY'
import datetime
import json
import sys
from pathlib import Path

payload = {
    "authorized": True,
    "scope": "CARoPE_locked_test_v2_stage_D",
    "run_class": "exploratory_gate_override",
    "requested_by": "user",
    "requested_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "reason": (
        "User explicitly requested Stage D evaluation despite validation_effect_gate.ok=false; "
        "results must not be represented as preregistered gate-passing evidence."
    ),
    "failed_gate_path": str(Path(sys.argv[2]).resolve()),
    "primary_candidate": "A3_full_carope",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  PYTHONPATH="${project_root}/src" "${python_bin}" "${project_root}/scripts/build_shared_test_landmarks.py" \
    --data-dir "${data_dir}" --split test --seed 1337 --block-size 128 --out "${landmarks}"

  PYTHONPATH="${project_root}/src" "${python_bin}" "${project_root}/scripts/build_paper_protocol_freeze_manifest.py" \
    --split-audit "${audit}" --data-dir "${data_dir}" \
    --diseases-yaml "${project_root}/docs/selected_diseases.yaml" \
    --checkpoint "A0=${train_root}/A0_sincos_control/horizon_retry1/checkpoints/best_val_horizon_auc.pt" \
    --checkpoint "A1=${train_root}/A1_car_rope_trunk/horizon/checkpoints/best_val_horizon_auc.pt" \
    --checkpoint "A2=${train_root}/A2_relative_horizon_query/horizon/checkpoints/best_val_horizon_auc.pt" \
    --checkpoint "A3=${train_root}/A3_full_carope/horizon/checkpoints/best_val_horizon_auc.pt" \
    --asset "medical_control_evaluator=${project_root}/src/semantic_delphi_ukb/evaluate_medical_control_tasks.py" \
    --asset "comparison_evaluator=${project_root}/src/semantic_delphi_ukb/compare_horizon_control_tasks.py" \
    --asset "locked_test_runner=${project_root}/scripts/run_locked_test_paper_protocol.py" \
    --asset "validation_consistency=${validation_root}/validation_consistency.json" \
    --asset "validation_official_aggregates=${validation_root}/validation_official_aggregates.json" \
    --asset "validation_effect_gate=${effect_gate}" \
    --asset "exploratory_gate_override=${authorization}" \
    --asset "vocab=${data_dir}/vocab/dynamic_token_vocab.csv" \
    --asset "shared_validation_landmarks=${validation_root}/val_shared_landmarks.json" \
    --asset "shared_test_landmarks=${landmarks}" \
    --require-asset validation_effect_gate \
    --require-asset exploratory_gate_override \
    --require-asset locked_test_runner \
    --exploratory-override "${authorization}" \
    --model-data-dir "A0=${data_dir}" --model-split-audit "A0=${audit}" \
    --model-data-dir "A1=${data_dir}" --model-split-audit "A1=${audit}" \
    --model-data-dir "A2=${data_dir}" --model-split-audit "A2=${audit}" \
    --model-data-dir "A3=${data_dir}" --model-split-audit "A3=${audit}" \
    --protocol-json "${protocol}" --test-eids-csv "${test_eids}" \
    --confirm-checkpoint-excludes-test --out "${freeze}"

  PYTHONPATH="${project_root}/src" "${python_bin}" "${project_root}/scripts/verify_paper_protocol_freeze_manifest.py" \
    --freeze-manifest "${freeze}" --out "${verification}"
}

write_status running
if run_prepare > "${log}" 2>&1; then
  write_status finished 0
else
  code=$?
  write_status failed "${code}"
  exit "${code}"
fi
