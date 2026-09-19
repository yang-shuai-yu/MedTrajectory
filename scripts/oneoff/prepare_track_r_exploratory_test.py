"""Render one-shot exploratory Track R test commands after freeze verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "RUN_TRACK_R_V2_1_EXPLORATORY_TEST_ONCE"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--test-landmarks", type=Path, required=True)
    p.add_argument("--test-feature-output", type=Path, required=True)
    p.add_argument("--test-output-root", type=Path, required=True)
    p.add_argument("--confirm", required=True)
    args = p.parse_args(argv)
    if args.confirm != CONFIRMATION:
        raise ValueError(f"--confirm must equal {CONFIRMATION}")
    if args.test_feature_output.exists() or args.test_output_root.exists():
        raise FileExistsError("one-shot test feature/output target already exists")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_track_r_freeze.py"), "--freeze", str(args.freeze)], cwd=ROOT, check=True)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8-sig"))
    landmarks = json.loads(args.test_landmarks.read_text(encoding="utf-8-sig"))
    if landmarks.get("split") != "test" or any("position" not in item for item in landmarks.get("landmarks", [])):
        raise ValueError("one-shot test requires a fixed-position test landmark manifest")
    commands = {
        "build_test_features": [
            sys.executable, str(ROOT / "scripts/build_track_r_landmark_features.py"),
            "--protocol", freeze["protocol"]["path"],
            "--data-dir", str(Path(freeze["data_manifest"]["path"]).parent),
            "--split", "test", "--landmark-manifest", str(args.test_landmarks),
            "--output", str(args.test_feature_output),
        ],
        "test_output_root": str(args.test_output_root),
        "models": freeze["models"],
        "result_class": "exploratory",
        "warning": "Do not select, tune, or relaunch models from these test results.",
    }
    print(json.dumps(commands, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
