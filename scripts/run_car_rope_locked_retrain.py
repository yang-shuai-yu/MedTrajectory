"""Validate, print, or execute the CARoPE retraining matrix on test-excluded train/val data."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.paper_protocol_audit import eid_set_sha256, read_eid_csv  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / "configs/paper_protocol_v1/CARoPE_locked_retrain_v2.json")
    p.add_argument("--stage", choices=("pretraining", "horizon", "all"), default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--variants",
        default="",
        help="comma-separated variant names; default runs the complete reviewed matrix",
    )
    p.add_argument("--validate-inputs", action="store_true")
    p.add_argument(
        "--allow-existing-output-root",
        action="store_true",
        help="permit a shared output root when every selected run directory is empty",
    )
    p.add_argument("--execute", action="store_true", help="run jobs sequentially; default only prints commands")
    return p


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_spec(spec: dict, require_inputs: bool) -> dict:
    if spec.get("data_policy") != "locked_test_excluded_train_val":
        raise ValueError("locked retraining must use test-excluded train/val data")
    if spec.get("seeds") != [42]:
        raise ValueError("reviewed minimum matrix must start with seed42 only")
    if [float(value) for value in spec.get("horizons_years", [])] != [1.0, 5.0, 10.0]:
        raise ValueError("locked protocol horizons must be 1,5,10 years")
    if [variant.get("name") for variant in spec.get("variants", [])] != [
        "A0_sincos_control", "A1_car_rope_trunk", "A2_relative_horizon_query", "A3_full_carope"
    ]:
        raise ValueError("locked protocol requires the reviewed A0-A3 matrix")
    paths = {name: _path(spec[name]) for name in ("data_dir", "split_audit", "test_eids_csv", "diseases_yaml")}
    existence = {name: path.exists() for name, path in paths.items()}
    if require_inputs and not all(existence.values()):
        missing = [name for name, exists in existence.items() if not exists]
        raise FileNotFoundError(f"locked retraining inputs are missing: {missing}")
    if all(existence.values()):
        data_manifest = json.loads((paths["data_dir"] / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
        split_audit = json.loads(paths["split_audit"].read_text(encoding="utf-8-sig"))
        expected_set_hash = eid_set_sha256(read_eid_csv(paths["test_eids_csv"]))
        if data_manifest.get("protocol_id") != spec.get("protocol_id"):
            raise ValueError("data manifest protocol_id does not match retraining spec")
        if data_manifest.get("locked_test_source", {}).get("eid_set_sha256") != expected_set_hash:
            raise ValueError("data manifest test EIDs do not match the frozen list")
        if not split_audit.get("disjoint", {}).get("ok") or not split_audit.get("expected_test_eids", {}).get("ok"):
            raise ValueError("split audit has not passed disjointness and frozen-EID checks")
    return {"paths": {name: str(path) for name, path in paths.items()}, "exists": existence}


def variant_command(spec: dict, variant: dict, seed: int, stage: str, device: str) -> list[str]:
    run_root = _path(spec["output_root"]) / f"seed{seed}" / variant["name"]
    run_dir = run_root / stage
    stage_config = spec[stage]
    entrypoint = spec["pretraining_entrypoint"] if stage == "pretraining" else spec["horizon_entrypoint"]
    command = [
        sys.executable, str(_path(entrypoint)),
        "--data-dir", str(_path(spec["data_dir"])),
        "--diseases-yaml", str(_path(spec["diseases_yaml"])),
        "--run-dir", str(run_dir),
        "--device", device,
        "--seed", str(seed),
        "--horizons", ",".join(str(value) for value in spec["horizons_years"]),
        "--max-iters", str(stage_config["max_iters"]),
        "--eval-interval", str(stage_config["eval_interval"]),
    ]
    for key in ("use_age_encoding", "use_age_rope", "use_relative_horizon_query", "time_gap_loss_weight"):
        command.extend([f"--{key.replace('_', '-')}", str(variant[key])])
    if stage == "horizon":
        command.extend(["--init-from-ckpt", str(run_root / "pretraining/checkpoints/best_val_loss.pt")])
    return command


def select_variants(spec: dict, requested: str) -> list[dict]:
    variants = spec["variants"]
    if not requested.strip():
        return variants
    by_name = {variant["name"]: variant for variant in variants}
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if len(names) != len(set(names)):
        raise ValueError("--variants must not contain duplicate names")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return [by_name[name] for name in names]


def validate_horizon_stage(spec: dict, variants: list[dict] | None = None) -> dict:
    variants = spec["variants"] if variants is None else variants
    missing_checkpoints = []
    occupied_outputs = []
    for seed in spec["seeds"]:
        for variant in variants:
            run_root = _path(spec["output_root"]) / f"seed{seed}" / variant["name"]
            checkpoint = run_root / "pretraining/checkpoints/best_val_loss.pt"
            if not checkpoint.is_file():
                missing_checkpoints.append(str(checkpoint))
            horizon_dir = run_root / "horizon"
            if horizon_dir.exists() and any(path.name != "stdout.log" for path in horizon_dir.iterdir()):
                occupied_outputs.append(str(horizon_dir))
    if missing_checkpoints or occupied_outputs:
        raise RuntimeError(
            "horizon-stage preflight failed: "
            f"missing_checkpoints={missing_checkpoints}, occupied_outputs={occupied_outputs}"
        )
    return {"pretraining_checkpoints": len(spec["seeds"]) * len(spec["variants"]), "horizon_outputs_empty": True}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    variants = select_variants(spec, args.variants)
    contract = validate_spec(spec, require_inputs=args.validate_inputs or args.execute)
    if args.stage == "horizon" and (args.validate_inputs or args.execute):
        contract["horizon_stage"] = validate_horizon_stage(spec, variants)
    stages = ("pretraining", "horizon") if args.stage == "all" else (args.stage,)
    jobs = [
        variant_command(spec, variant, int(seed), stage, args.device)
        for seed in spec["seeds"]
        for variant in variants
        for stage in stages
    ]
    output_root = _path(spec["output_root"])
    if args.execute and args.stage in ("pretraining", "all"):
        if output_root.exists() and not args.allow_existing_output_root:
            raise FileExistsError(f"retraining output root already exists: {output_root}")
        if args.allow_existing_output_root:
            occupied = []
            for seed in spec["seeds"]:
                for variant in variants:
                    run_root = output_root / f"seed{seed}" / variant["name"]
                    for stage in stages:
                        stage_dir = run_root / stage
                        if stage_dir.exists() and any(path.name != "stdout.log" for path in stage_dir.iterdir()):
                            occupied.append(str(stage_dir))
            if occupied:
                raise FileExistsError(f"selected retraining run directories are not empty: {occupied}")
    print(json.dumps({
        "execute": args.execute,
        "stage": args.stage,
        "variants": [variant["name"] for variant in variants],
        "input_contract": contract,
        "output_root": str(output_root),
        "commands": [shlex.join(command) for command in jobs],
    }, indent=2))
    if not args.execute:
        return 0
    for command in jobs:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
