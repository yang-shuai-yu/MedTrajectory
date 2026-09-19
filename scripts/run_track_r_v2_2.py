"""Render Track R v2.2 commands; execute only with an explicit flag."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/track_r_v2_2/TRACK_R_v2_2.json",
    )
    value.add_argument("--lane", choices=("wavelengths", "train", "evaluate", "all"), default="all")
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="SEED:MODEL",
        help="restrict the train lane to exact seed:model pairs; repeat as needed",
    )
    value.add_argument("--execute", action="store_true")
    return value


def command(*parts) -> list[str]:
    return [str(part) for part in parts]


def paths(protocol: dict, seed: Optional[int] = None) -> dict[str, Path]:
    output = ROOT / protocol["output_root"]
    result = {
        "data": Path(protocol["output_data_dir"]),
        "output": output,
        "wavelengths": ROOT / protocol["wavelength_contract"]["manifest"],
        "landmarks": ROOT / protocol["shared_validation_landmark_manifest"],
    }
    if seed is not None:
        result["run"] = output / "runs" / protocol["run_tag_template"].format(seed=seed)
    return result


def wavelength_jobs(protocol_path: Path, protocol: dict) -> list[list[str]]:
    value = paths(protocol)
    return [command(
        sys.executable,
        ROOT / "scripts/build_track_r_v2_2_wavelengths.py",
        "--protocol", protocol_path,
        "--data-dir", value["data"],
        "--output", value["wavelengths"],
    )]


def _model_args(protocol: dict, variant: dict, value: dict[str, Path], device: str, seed: int) -> list[str]:
    architecture = protocol["architecture"]
    args = [
        "--track-r-protocol", protocol["_path"],
        "--data-dir", value["data"],
        "--diseases-yaml", ROOT / protocol["diseases_yaml"],
        "--device", device,
        "--seed", seed,
        "--block-size", protocol["dynamic_context_length"],
        "--n-layer", architecture["n_layer"],
        "--n-head", architecture["n_head"],
        "--n-embd", architecture["n_embd"],
        "--include-static-prefix", "true",
        "--use-age-encoding", str(variant["use_age_encoding"]).lower(),
        "--use-age-rope", str(variant["use_age_rope"]).lower(),
        "--age-rope-variant", variant["age_rope_variant"],
        "--use-relative-horizon-query", "true",
        "--horizons", ",".join(str(item) for item in protocol["horizons_years"]),
    ]
    if variant["age_rope_variant"] == "additive_v2_2":
        args.extend(["--rope-wavelengths-manifest", value["wavelengths"]])
    return [str(item) for item in args]


def parse_pairs(specs: list[str], protocol: dict) -> set[tuple[int, str]]:
    allowed_seeds = {int(seed) for seed in protocol["seeds"]}
    allowed_models = {variant["name"] for variant in protocol["transformer_variants"]}
    pairs = set()
    for spec in specs:
        raw_seed, separator, model = spec.partition(":")
        if not separator or not raw_seed.isdigit():
            raise ValueError(f"invalid --pair {spec!r}; expected SEED:MODEL")
        pair = (int(raw_seed), model)
        if pair[0] not in allowed_seeds or pair[1] not in allowed_models:
            raise ValueError(f"unregistered Track R v2.2 pair: {spec}")
        if pair in pairs:
            raise ValueError(f"duplicate --pair: {spec}")
        pairs.add(pair)
    return pairs


def training_jobs(
    protocol_path: Path,
    protocol: dict,
    device: str,
    pairs: Optional[set[tuple[int, str]]] = None,
) -> list[list[str]]:
    protocol = dict(protocol)
    protocol["_path"] = protocol_path
    jobs = []
    budget = protocol["training_budget"]
    for seed in protocol["seeds"]:
        value = paths(protocol, int(seed))
        for variant in protocol["transformer_variants"]:
            model = variant["name"]
            if pairs and (int(seed), model) not in pairs:
                continue
            common = _model_args(protocol, variant, value, device, int(seed))
            pretraining = value["run"] / "pretraining" / model
            risk = value["run"] / "risk" / model
            jobs.append(command(
                sys.executable,
                ROOT / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
                "--run-dir", pretraining,
                "--max-iters", budget["pretraining_max_iters"],
                "--time-gap-loss-weight", budget["pretraining_time_gap_loss_weight"],
                *common,
            ))
            jobs.append(command(
                sys.executable,
                ROOT / "src/semantic_delphi_ukb/train_car_rope.py",
                "--run-dir", risk,
                "--init-from-ckpt", pretraining / "checkpoints/last.pt",
                "--max-iters", budget["risk_max_iters"],
                "--next-event-loss-weight", budget["risk_next_event_loss_weight"],
                "--horizon-risk-loss-weight", budget["risk_horizon_loss_weight"],
                "--time-gap-loss-weight", budget["risk_time_gap_loss_weight"],
                *common,
            ))
    return jobs


def evaluation_jobs(protocol_path: Path, protocol: dict, device: str) -> list[list[str]]:
    jobs = []
    for seed in protocol["seeds"]:
        value = paths(protocol, int(seed))
        for variant in protocol["transformer_variants"]:
            model = variant["name"]
            pretraining = value["run"] / "pretraining" / model
            risk = value["run"] / "risk" / model
            jobs.append(command(
                sys.executable,
                "-m", "semantic_delphi_ukb.evaluate_track_r_v2_2_pretraining",
                "--protocol", protocol_path,
                "--data-dir", value["data"],
                "--checkpoint", pretraining / "checkpoints/last.pt",
                "--model-name", model,
                "--split", "val",
                "--out-dir", value["run"] / "pretraining_fidelity" / model,
                "--device", device,
            ))
            jobs.append(command(
                sys.executable,
                "-m", "semantic_delphi_ukb.evaluate_track_r",
                "--family", "carope",
                "--model-name", model,
                "--checkpoint", risk / "checkpoints/last.pt",
                "--protocol", protocol_path,
                "--data-dir", value["data"],
                "--split", "val",
                "--landmark-manifest", value["landmarks"],
                "--out-dir", value["run"] / "clinical" / model,
                "--include-static-prefix", "true",
                "--device", device,
            ))
    jobs.append(command(
        sys.executable,
        "-m", "semantic_delphi_ukb.finalize_track_r_v2_2",
        "--protocol", protocol_path,
        "--out-dir", paths(protocol)["output"] / "final_assessment",
    ))
    return jobs


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    if args.pair and args.lane != "train":
        raise ValueError("--pair is valid only with --lane train")
    pairs = parse_pairs(args.pair, protocol)
    lanes = {
        "wavelengths": wavelength_jobs(args.protocol, protocol),
        "train": training_jobs(args.protocol, protocol, args.device, pairs or None),
        "evaluate": evaluation_jobs(args.protocol, protocol, args.device),
    }
    selected = list(lanes) if args.lane == "all" else [args.lane]
    print(json.dumps({
        "execute": args.execute,
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "locked_test_read": protocol["locked_test_read"],
        "pairs": sorted(f"{seed}:{model}" for seed, model in pairs),
        "lanes": {name: [shlex.join(job) for job in lanes[name]] for name in selected},
        "test_commands_included": False,
    }, indent=2))
    if not args.execute:
        return 0
    for name in selected:
        for job in lanes[name]:
            subprocess.run(job, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
