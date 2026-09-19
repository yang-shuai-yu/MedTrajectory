"""Run all validation-only Track G death-hazard heads sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results/track_g_v1/val_death_hazard_v2_followup"
MODELS = ("A0", "A2", "ETHOS-Matched", "Foresight-Matched")
SEEDS = (42, 43, 44)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command(model: str, seed: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/train_track_g_death_hazard_head.py"),
        "--model", model,
        "--seed", str(seed),
        "--run-dir", str(OUTPUT_ROOT / f"seed{seed}" / model),
        "--device", "cuda",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = [(model, seed, command(model, seed)) for seed in SEEDS for model in MODELS]
    if not args.execute:
        print(json.dumps({"jobs": [job for _, _, job in jobs]}, indent=2))
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
    }
    atomic_json(status_path, status)
    try:
        for index, (model, seed, job) in enumerate(jobs, start=1):
            summary = OUTPUT_ROOT / f"seed{seed}" / model / "summary.json"
            if summary.is_file():
                status["completed"] = index
                atomic_json(status_path, status)
                continue
            status["current"] = {"model": model, "seed": seed}
            atomic_json(status_path, status)
            subprocess.run(job, cwd=ROOT, check=True)
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
