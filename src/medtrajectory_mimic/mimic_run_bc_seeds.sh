#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Three-seed completion for the remaining representations.
#   1) Bm matched multitype (Experiment 1, matched patients): bm_m{1,2,3}_a0_s{43,44}
#   2) Design B  Experiment 2 (A0 vs A2): vB_m3_a{0,2}_s{43,44}
#   3) Design C  Experiment 2 (A0 vs A2): vC_a{0,2}_s{43,44}
# Trained as batches of two (measured fastest per model on this single GPU).
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_bc_seeds
mkdir -p "$LOGS"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0

train () {
  local name="$1" ddir="$2" proto="$3" seed="$4" rope="$5" variant="$6" extra="$7"
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/$ddir" --track-r-protocol "$DATA/$proto" \
    --include-static-prefix true --age-rope-variant "$variant" \
    --use-age-encoding true --use-age-rope "$rope" \
    $extra \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed "$seed" --no-tensorboard > "$LOGS/$name.log" 2>&1
}

batch () {
  echo "=== batch: $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  local pids=()
  for spec in "$@"; do
    IFS='|' read -r name ddir proto seed rope variant extra <<< "$spec"
    train "$name" "$ddir" "$proto" "$seed" "$rope" "$variant" "$extra" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== batch done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

PBM=$DATA/visit_Bm_protocol.json
PB=$DATA/visit_B_m3_protocol.json
PC=$DATA/visit_C_protocol.json
WB=$DATA/visit_B_m3_wavelengths.json
WC=$DATA/visit_C_wavelengths.json

# ---- Bm matched multitype, seeds 43 and 44 ----
batch "bm_m1_a0_s43|visit_Bm_m1_trackr|visit_Bm_protocol.json|43|false|legacy|" \
      "bm_m2_a0_s43|visit_Bm_m2_trackr|visit_Bm_protocol.json|43|false|legacy|"
batch "bm_m3_a0_s43|visit_Bm_m3_trackr|visit_Bm_protocol.json|43|false|legacy|" \
      "bm_m1_a0_s44|visit_Bm_m1_trackr|visit_Bm_protocol.json|44|false|legacy|"
batch "bm_m2_a0_s44|visit_Bm_m2_trackr|visit_Bm_protocol.json|44|false|legacy|" \
      "bm_m3_a0_s44|visit_Bm_m3_trackr|visit_Bm_protocol.json|44|false|legacy|"

# ---- Design B Experiment 2, seeds 43 and 44 ----
batch "vB_m3_a0_s43|visit_B_m3_trackr|visit_B_m3_protocol.json|43|false|legacy|" \
      "vB_m3_a2_s43|visit_B_m3_trackr|visit_B_m3_protocol.json|43|true|additive_v2_2|--rope-wavelengths-manifest $WB"
batch "vB_m3_a0_s44|visit_B_m3_trackr|visit_B_m3_protocol.json|44|false|legacy|" \
      "vB_m3_a2_s44|visit_B_m3_trackr|visit_B_m3_protocol.json|44|true|additive_v2_2|--rope-wavelengths-manifest $WB"

# ---- Design C Experiment 2, seeds 43 and 44 ----
batch "vC_a0_s43|visit_C_trackr|visit_C_protocol.json|43|false|legacy|" \
      "vC_a2_s43|visit_C_trackr|visit_C_protocol.json|43|true|additive_v2_2|--rope-wavelengths-manifest $WC"
batch "vC_a0_s44|visit_C_trackr|visit_C_protocol.json|44|false|legacy|" \
      "vC_a2_s44|visit_C_trackr|visit_C_protocol.json|44|true|additive_v2_2|--rope-wavelengths-manifest $WC"

echo "ALL BC 3-SEED TRAINING DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
