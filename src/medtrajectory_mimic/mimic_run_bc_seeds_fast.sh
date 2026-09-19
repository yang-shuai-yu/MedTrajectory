#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Remaining six models of the B/C three-seed completion, re-queued after the
# thread-thrash fix.
#
# Diagnosis (bench_clean.py, idle machine, 2026-09-13):
#   * torch.get_num_threads() defaulted to 104 (nproc reports 208, the cgroup
#     grants 20 cores) and torch.get_num_interop_threads() to 208
#   * per training step: 18.9 ms of real work, but 122 ms of CPU burned at 16
#     threads and ~750 ms at the default; wall time is flat across thread counts
#     -> essentially all CPU was thread-pool spin
#   * cpu.stat showed 93.7% of cgroup periods throttled
#   * batch assembly is bit-identical from 1 to 104 threads (sha256 certificate)
#   * pure GPU floor: 12.91 ms/step; one process at OMP_NUM_THREADS=1 runs at
#     18.9 ms/step (69% GPU utilisation) instead of the observed 101 ms/step
#
# Therefore OMP_NUM_THREADS=1 and more concurrent processes.  The GPU caps
# aggregate throughput at ~77 steps/s, so 4+2 and 3+3 and 6-at-once all finish
# in about the same wall time; the waves below keep each A0/A2 pair together.
set -uo pipefail
REPO="${MEDTRAJECTORY_ROOT}"
DATA="${MEDTRAJECTORY_MIMIC_ROOT}"
PY="${MEDTRAJECTORY_PYTHON:-python3}"
YAML=$DATA/multitype/mimic_diseases.yaml
LOGS=$DATA/logs_bc_seeds_fast
mkdir -p "$LOGS"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

train () {
  local name="$1" ddir="$2" proto="$3" seed="$4" rope="$5" variant="$6" extra="$7"
  # these two directories hold partial checkpoints from the killed thread-thrashed run
  if [ "$name" = "vB_m3_a0_s44" ] || [ "$name" = "vB_m3_a2_s44" ]; then
    rm -rf "$DATA/runs/$name"
  fi
  "$PY" src/semantic_delphi_ukb/train_car_rope_pretraining.py \
    --data-dir "$DATA/$ddir" --track-r-protocol "$DATA/$proto" \
    --include-static-prefix true --age-rope-variant "$variant" \
    --use-age-encoding true --use-age-rope "$rope" \
    $extra \
    --diseases-yaml "$YAML" \
    --run-dir "$DATA/runs/$name" \
    --device cuda --max-iters 100000 --seed "$seed" --no-tensorboard > "$LOGS/$name.log" 2>&1
  echo "  [$(date -u +%H:%M:%SZ)] $name exit=$?"
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

PB=$DATA/visit_B_m3_protocol.json
PC=$DATA/visit_C_protocol.json
WB=$DATA/visit_B_m3_wavelengths.json
WC=$DATA/visit_C_wavelengths.json

# ---- single wave: all six remaining models (GPU-bound, so 4+2 and 6-at-once
#      take the same wall time; one wave avoids any idle GPU between waves) ----
batch "vB_m3_a0_s44|visit_B_m3_trackr|visit_B_m3_protocol.json|44|false|legacy|" \
      "vB_m3_a2_s44|visit_B_m3_trackr|visit_B_m3_protocol.json|44|true|additive_v2_2|--rope-wavelengths-manifest $WB" \
      "vC_a0_s43|visit_C_trackr|visit_C_protocol.json|43|false|legacy|" \
      "vC_a2_s43|visit_C_trackr|visit_C_protocol.json|43|true|additive_v2_2|--rope-wavelengths-manifest $WC" \
      "vC_a0_s44|visit_C_trackr|visit_C_protocol.json|44|false|legacy|" \
      "vC_a2_s44|visit_C_trackr|visit_C_protocol.json|44|true|additive_v2_2|--rope-wavelengths-manifest $WC"

echo "ALL BC 3-SEED TRAINING DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- evaluation + summaries, also single-threaded ----
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash "$DATA/mimic_run_eval_bc_seeds.sh"
echo "ALL BC 3-SEED EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
