"""Render or execute the one-shot Track G v1 locked-test pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_g_contract import (  # noqa: E402
    assert_split_allowed,
    assert_training_allowed,
    load_track_g_protocol,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=ROOT / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument("--device", default="cuda")
    value.add_argument("--execute", action="store_true")
    value.add_argument("--format", choices=("json", "shell"), default="json")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_test_jobs(protocol_path: Path, protocol: dict, device: str) -> list[list[str]]:
    python = sys.executable
    output_root = ROOT / protocol["output_root"]
    cohort = ROOT / protocol["cohort_manifests"]["locked_test"]
    sampler = output_root / "manifests/sampler_selection.json"
    jobs = [[
        python,
        str(ROOT / "src/semantic_delphi_ukb/preflight_track_g.py"),
        "--protocol", str(protocol_path),
        "--require-trained-baselines",
    ], [
        python,
        str(ROOT / "src/semantic_delphi_ukb/build_track_g_cohort.py"),
        "--protocol", str(protocol_path),
        "--split", "test",
        "--output", str(cohort),
    ]]
    for seed in protocol["seeds"]:
        for model in protocol["models"]:
            checkpoint = ROOT / model["checkpoint"].format(seed=seed)
            jobs.append([
                python,
                str(ROOT / "src/semantic_delphi_ukb/evaluate_track_g_generation.py"),
                "--protocol", str(protocol_path),
                "--model", model["name"],
                "--seed", str(seed),
                "--checkpoint", str(checkpoint),
                "--split", "test",
                "--out-dir", str(output_root / "test" / f"seed{seed}" / model["name"]),
                "--device", device,
                "--sampler-manifest", str(sampler),
                "--cohort-manifest", str(cohort),
            ])
    jobs.append([
        python,
        str(ROOT / "src/semantic_delphi_ukb/finalize_track_g.py"),
        "--protocol", str(protocol_path),
        "--split", "test",
        "--input-root", str(output_root / "test"),
        "--out-dir", str(output_root / "test_assessment"),
    ])
    return jobs


def assert_one_shot_outputs_absent(protocol: dict) -> None:
    output_root = ROOT / protocol["output_root"]
    paths = (
        ROOT / protocol["cohort_manifests"]["locked_test"],
        output_root / "test",
        output_root / "test_assessment",
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Track G locked test is one-shot; outputs already exist: {existing}")


def verify_frozen_validation_artifacts(protocol: dict) -> None:
    authorization = protocol["locked_test_authorization"]
    output_root = ROOT / protocol["output_root"]
    expected = {
        output_root / "manifests/sampler_selection.json": authorization["sampler_selection_sha256"],
        output_root / "val_assessment/assessment.json": authorization["validation_assessment_sha256"],
    }
    for path, expected_hash in expected.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"frozen validation artifact hash mismatch: {path}")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_g_protocol(args.protocol, ROOT)
    assert_training_allowed(protocol)
    assert_split_allowed(protocol, "test")
    jobs = locked_test_jobs(args.protocol, protocol, args.device)
    rendered = [shlex.join(job) for job in jobs]
    payload = {
        "execute": args.execute,
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "validation_protocol_sha256": protocol["locked_test_authorization"]["validation_protocol_sha256"],
        "one_shot": True,
        "job_count": len(jobs),
        "jobs": rendered,
    }
    if args.format == "shell":
        print("\n".join(rendered))
    else:
        print(json.dumps(payload, indent=2))
    if not args.execute:
        return 0
    assert_one_shot_outputs_absent(protocol)
    verify_frozen_validation_artifacts(protocol)
    for job in jobs:
        subprocess.run(job, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
