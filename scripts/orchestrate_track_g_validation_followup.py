"""Wait for no-death validation, then run paired analysis and death heads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/track_g_v1"
STATUS_PATH = RESULT_ROOT / "validation_followup_status.json"


def atomic_status(payload: dict) -> None:
    temporary = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
    payload = {**payload, "updated_at": datetime.now(timezone.utc).isoformat()}
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS_PATH)


def run(*parts: str) -> None:
    subprocess.run([sys.executable, *parts], cwd=ROOT, check=True)


def main() -> int:
    state = {"status": "running", "phase": "waiting_no_death_validation"}
    atomic_status(state)
    try:
        queue_status = RESULT_ROOT / "val_no_death_token/queue_status.json"
        while True:
            if queue_status.is_file():
                payload = json.loads(queue_status.read_text(encoding="utf-8"))
                state.update({
                    "no_death_status": payload.get("status"),
                    "no_death_completed": payload.get("completed", 0),
                    "no_death_total": payload.get("total", 6),
                })
                atomic_status(state)
                if payload.get("status") == "finished":
                    break
                if payload.get("status") == "failed":
                    raise RuntimeError(f"no-death validation failed: {payload}")
            time.sleep(30)

        state["phase"] = "paired_no_death_assessment"
        atomic_status(state)
        run("scripts/analyze_track_g_no_death_validation.py")

        state["phase"] = "death_hazard_queue"
        atomic_status(state)
        run("scripts/run_track_g_death_hazard_validation.py", "--execute")

        state["phase"] = "death_hazard_assessment"
        atomic_status(state)
        run("scripts/analyze_track_g_death_hazard_validation.py")

        state.update({"status": "finished", "phase": "finished"})
        atomic_status(state)
    except BaseException as exc:
        state.update({
            "status": "failed",
            "error": type(exc).__name__,
            "message": str(exc),
        })
        atomic_status(state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
