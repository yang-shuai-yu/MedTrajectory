"""Validate, print, or execute the A1 capacity-scaling matrix."""

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


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / "configs/paper_protocol_v1/CARoPE_capacity_scaling_v1.json")
    p.add_argument("--stage", choices=("pretraining", "horizon", "all"), default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--variants", default="")
    p.add_argument("--validate-inputs", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p


def validate_spec(spec: dict, require_inputs: bool) -> None:
    if spec.get("status") != "exploratory_capacity_study":
        raise ValueError("capacity results must be labelled exploratory_capacity_study")
    if spec.get("data_policy") != "locked_test_excluded_train_val":
        raise ValueError("capacity training must use test-excluded train/val data")
    if spec.get("seeds") != [42] or spec.get("horizons_years") != [1.0, 5.0, 10.0]:
        raise ValueError("capacity protocol requires seed42 and 1/5/10-year horizons")
    variants = spec.get("variants", [])
    if [item.get("name") for item in variants] != ["A1-M", "A1-L"]:
        raise ValueError("capacity protocol requires A1-M and A1-L")
    for item in variants:
        if item["n_layer"] != 12 or item["n_embd"] % item["n_head"]:
            raise ValueError(f"invalid Transformer dimensions for {item['name']}")
        if any(item[key] != expected for key, expected in {
            "use_age_encoding": "false", "use_age_rope": "true",
            "use_relative_horizon_query": "false", "time_gap_loss_weight": 0.0,
        }.items()):
            raise ValueError(f"{item['name']} is not an A1-only capacity variant")
    if not require_inputs:
        return
    paths = {name: _path(spec[name]) for name in ("data_dir", "split_audit", "test_eids_csv", "diseases_yaml")}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"capacity inputs are missing: {missing}")
    manifest = json.loads((paths["data_dir"] / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    audit = json.loads(paths["split_audit"].read_text(encoding="utf-8-sig"))
    expected_hash = eid_set_sha256(read_eid_csv(paths["test_eids_csv"]))
    if manifest.get("protocol_id") != spec["protocol_id"]:
        raise ValueError("data manifest protocol_id mismatch")
    if manifest.get("locked_test_source", {}).get("eid_set_sha256") != expected_hash:
        raise ValueError("frozen test EID hash mismatch")
    if not audit.get("disjoint", {}).get("ok") or not audit.get("expected_test_eids", {}).get("ok"):
        raise ValueError("split audit did not pass")


def select_variants(spec: dict, requested: str) -> list[dict]:
    if not requested.strip():
        return spec["variants"]
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if len(names) != len(set(names)):
        raise ValueError("--variants contains duplicates")
    by_name = {item["name"]: item for item in spec["variants"]}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return [by_name[name] for name in names]


def variant_command(spec: dict, variant: dict, stage: str, device: str, require_checkpoint: bool = False) -> list[str]:
    run_root = _path(spec["output_root"]) / "seed42" / variant["name"]
    stage_spec = spec[stage]
    entrypoint = spec["pretraining_entrypoint"] if stage == "pretraining" else spec["horizon_entrypoint"]
    command = [
        sys.executable, str(_path(entrypoint)),
        "--data-dir", str(_path(spec["data_dir"])),
        "--diseases-yaml", str(_path(spec["diseases_yaml"])),
        "--run-dir", str(run_root / stage), "--device", device, "--seed", "42",
        "--horizons", ",".join(str(value) for value in spec["horizons_years"]),
        "--max-iters", str(stage_spec["max_iters"]), "--eval-interval", str(stage_spec["eval_interval"]),
        "--n-layer", str(variant["n_layer"]), "--n-head", str(variant["n_head"]),
        "--n-embd", str(variant["n_embd"]),
    ]
    for key in ("use_age_encoding", "use_age_rope", "use_relative_horizon_query", "time_gap_loss_weight"):
        command.extend([f"--{key.replace('_', '-')}", str(variant[key])])
    if stage == "horizon":
        checkpoint = run_root / "pretraining/checkpoints/best_val_loss.pt"
        if require_checkpoint and not checkpoint.is_file():
            raise FileNotFoundError(f"missing validation-selected pretraining checkpoint: {checkpoint}")
        command.extend(["--init-from-ckpt", str(checkpoint)])
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    validate_spec(spec, args.validate_inputs or args.execute)
    variants = select_variants(spec, args.variants)
    stages = ("pretraining", "horizon") if args.stage == "all" else (args.stage,)
    jobs = [variant_command(spec, variant, stage, args.device) for variant in variants for stage in stages]
    print(json.dumps({
        "experiment_label": "exploratory_capacity_study",
        "variants": [item["name"] for item in variants],
        "stages": list(stages),
        "commands": [shlex.join(command) for command in jobs],
    }, indent=2))
    if not args.execute:
        return 0
    for variant in variants:
        run_root = _path(spec["output_root"]) / "seed42" / variant["name"]
        for stage in stages:
            run_dir = run_root / stage
            if run_dir.exists() and any(path.name != "stdout.log" for path in run_dir.iterdir()):
                raise FileExistsError(f"run directory is not empty: {run_dir}")
    for variant in variants:
        for stage in stages:
            command = variant_command(spec, variant, stage, args.device, require_checkpoint=True)
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
