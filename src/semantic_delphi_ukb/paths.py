"""Paths for the MedTrajectory UKB pipeline.

All cluster-specific locations are resolved from environment variables so the
code runs outside the original compute allocation. Set at least
MEDTRAJECTORY_DATA_ROOT; everything else falls back to a sensible default
relative to this repository.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


#: this repository
REPO_ROOT = _env_path("MEDTRAJECTORY_ROOT", Path(__file__).resolve().parents[2])

#: parent of the repository on the original host (used by a few legacy log paths)
WORKSPACE_ROOT = _env_path("MEDTRAJECTORY_WORKSPACE_ROOT", REPO_ROOT.parent)

#: root holding prepared datasets; contains ``data/`` and the paper-protocol dirs
DATA_ROOT = _env_path("MEDTRAJECTORY_DATA_ROOT", REPO_ROOT / "data")

#: root holding run outputs
RUNS_ROOT = _env_path("MEDTRAJECTORY_RUNS_ROOT", REPO_ROOT / "results")

#: MIMIC-IV project root (prepared sequences, vocab, runs)
MIMIC_ROOT = _env_path("MEDTRAJECTORY_MIMIC_ROOT", REPO_ROOT / "mimic-iv")

#: third-party / other-project artefacts (reference workspaces, raw extracts).
#: No code path requires this to exist; it is only used by reference scripts.
EXTERNAL_ROOT = _env_path("MEDTRAJECTORY_EXTERNAL_ROOT", REPO_ROOT / "external")

#: interpreter used by shell launchers
PYTHON_BIN = os.environ.get("MEDTRAJECTORY_PYTHON", "python3")


def describe() -> str:
    """Human-readable resolution report; useful when a path looks wrong."""
    return "\n".join(
        [
            "MedTrajectory path resolution",
            "  REPO_ROOT      = %s" % REPO_ROOT,
            "  WORKSPACE_ROOT = %s" % WORKSPACE_ROOT,
            "  DATA_ROOT      = %s" % DATA_ROOT,
            "  RUNS_ROOT      = %s" % RUNS_ROOT,
            "  MIMIC_ROOT     = %s" % MIMIC_ROOT,
            "  EXTERNAL_ROOT  = %s" % EXTERNAL_ROOT,
            "  PYTHON_BIN     = %s" % PYTHON_BIN,
        ]
    )


if __name__ == "__main__":
    print(describe())
