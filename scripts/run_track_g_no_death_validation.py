"""Run the validation-only no-death-token trajectory ablation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/track_g_v1/TRACK_G_v1.json"
OUTPUT_ROOT = ROOT / "results/track_g_v1/val_no_death_token"
MODELS = ("A0", "A2")
SEEDS = (42, 43, 44)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_path(model: str, seed: int) -> Path:
    return (
        ROOT
        / f"results/track_r_v2_2/runs/seed{seed}_additiverope1/pretraining/{model}/checkpoints/last.pt"
    )


def evaluation_command(model: str, seed: int, max_patients: int | None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "src/semantic_delphi_ukb/evaluate_track_g_generation.py"),
        "--protocol",
        str(PROTOCOL),
        "--model",
        model,
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint_path(model, seed)),
        "--split",
        "val",
        "--out-dir",
        str(OUTPUT_ROOT / f"seed{seed}" / model),
        "--device",
        "cuda",
        "--sampler-manifest",
        str(ROOT / "results/track_g_v1/manifests/sampler_selection.json"),
        "--cohort-manifest",
        str(ROOT / "results/track_g_v1/manifests/val_generation_cohort.json"),
        "--exclude-death-token",
    ]
    if max_patients is not None:
        command.extend(("--max-patients", str(max_patients)))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = [evaluation_command(model, seed, args.max_patients) for seed in SEEDS for model in MODELS]
    if not args.execute:
        print(json.dumps({"jobs": jobs}, indent=2))
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_ROOT / "queue_status.json"
    status = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": 0,
        "total": len(jobs),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "max_patients": args.max_patients,
    }
    atomic_json(status_path, status)
    try:
        for index, command in enumerate(jobs, start=1):
            status["current"] = {"seed": SEEDS[(index - 1) // len(MODELS)], "model": MODELS[(index - 1) % len(MODELS)]}
            atomic_json(status_path, status)
            subprocess.run(command, cwd=ROOT, check=True)
            status["completed"] = index
            atomic_json(status_path, status)
    except BaseException as exc:
        status.update({"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        atomic_json(status_path, status)
        raise
    status.update({"status": "finished", "finished_at": datetime.now(timezone.utc).isoformat()})
    status.pop("current", None)
    atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
