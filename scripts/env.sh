#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MedTrajectory environment resolution for shell launchers.
#
# Source this at the top of a launcher:
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Every variable may be overridden from the outside; the defaults assume the
# repository layout (see src/semantic_delphi_ukb/paths.py, which mirrors this).
# ---------------------------------------------------------------------------

# resolve the repository root from this file's location
_MEDTRAJ_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MEDTRAJECTORY_ROOT:=$(cd "${_MEDTRAJ_ENV_DIR}/.." && pwd)}"
export MEDTRAJECTORY_ROOT

: "${MEDTRAJECTORY_WORKSPACE_ROOT:=$(cd "${MEDTRAJECTORY_ROOT}/.." && pwd)}"
: "${MEDTRAJECTORY_DATA_ROOT:=${MEDTRAJECTORY_ROOT}/data}"
: "${MEDTRAJECTORY_RUNS_ROOT:=${MEDTRAJECTORY_ROOT}/results}"
: "${MEDTRAJECTORY_MIMIC_ROOT:=${MEDTRAJECTORY_ROOT}/mimic-iv}"
: "${MEDTRAJECTORY_EXTERNAL_ROOT:=${MEDTRAJECTORY_ROOT}/external}"
: "${MEDTRAJECTORY_PYTHON:=python3}"
export MEDTRAJECTORY_WORKSPACE_ROOT MEDTRAJECTORY_DATA_ROOT MEDTRAJECTORY_RUNS_ROOT
export MEDTRAJECTORY_MIMIC_ROOT MEDTRAJECTORY_EXTERNAL_ROOT MEDTRAJECTORY_PYTHON

# convenience aliases used by the launchers
REPO_ROOT="${MEDTRAJECTORY_ROOT}"
DATA_ROOT="${MEDTRAJECTORY_DATA_ROOT}"
RUNS_ROOT="${MEDTRAJECTORY_RUNS_ROOT}"
MIMIC_ROOT="${MEDTRAJECTORY_MIMIC_ROOT}"
EXTERNAL_ROOT="${MEDTRAJECTORY_EXTERNAL_ROOT}"
PYTHON_BIN="${MEDTRAJECTORY_PYTHON}"
export REPO_ROOT DATA_ROOT RUNS_ROOT MIMIC_ROOT EXTERNAL_ROOT PYTHON_BIN
