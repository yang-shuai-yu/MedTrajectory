"""Prepare or execute the CARoPE validation-only component sweep.

The default is a dry run. ``--execute`` is required before any training
process is launched. This script never accepts a test split as an input.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/launch CARoPE validation-only variants.")
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/paper_protocol_v1/CARoPE_validation_sweep.json")
    parser.add_argument("--execute", action="store_true", help="actually launch training; omitted means dry run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    return parser


def variant_command(spec: dict, variant: dict, args: argparse.Namespace, stage: str, init_ckpt: Path | None = None) -> list[str]:
    output_root = ROOT / spec["output_root"]
    run_dir = output_root / variant["name"] / stage
    entrypoint = spec["pretraining_entrypoint"] if stage == "pretraining" else spec["entrypoint"]
    command = [
        sys.executable,
        str(ROOT / entrypoint),
        "--data-dir", str(ROOT / spec["data_dir"]),
        "--diseases-yaml", str(ROOT / spec["diseases_yaml"]),
        "--run-dir", str(run_dir),
        "--device", args.device,
        "--seed", str(spec["seed"]),
        "--horizons", ",".join(str(value) for value in spec["horizons_years"]),
    ]
    for key in ("use_age_encoding", "use_age_rope", "use_relative_horizon_query", "time_gap_loss_weight"):
        command.extend([f"--{key.replace('_', '-')}", str(variant[key])])
    if init_ckpt is not None:
        command.extend(["--init-from-ckpt", str(init_ckpt)])
    if args.max_iters is not None:
        command.extend(["--max-iters", str(args.max_iters)])
    if args.eval_interval is not None:
        command.extend(["--eval-interval", str(args.eval_interval)])
    return command


def main() -> int:
    args = build_parser().parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("data_policy") != "train_val_only_until_gate":
        raise ValueError("CARoPE sweep must be train/validation-only")
    jobs = []
    for variant in spec["variants"]:
        pretrain_dir = ROOT / spec["output_root"] / variant["name"] / "pretraining"
        pretrain_ckpt = pretrain_dir / "checkpoints" / "best_val_loss.pt"
        jobs.append((variant, "pretraining", variant_command(spec, variant, args, "pretraining")))
        jobs.append((variant, "horizon", variant_command(spec, variant, args, "horizon", pretrain_ckpt)))
    print(json.dumps({"execute": args.execute, "commands": [shlex.join(cmd) for _, _, cmd in jobs]}, indent=2))
    if not args.execute:
        return 0
    for _, _, command in jobs:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
