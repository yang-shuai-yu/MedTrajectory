"""Render the review-gated Track G v1 pipeline; execute only with --execute inside tmux."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_g_contract import assert_training_allowed, load_track_g_protocol  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=ROOT / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument(
        "--lane",
        choices=("preflight", "cohort", "train", "sampler-grid", "sampler-select", "validate", "finalize", "all"),
        default="all",
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--data-dir", type=Path, default=None)
    value.add_argument("--pair", action="append", default=[], metavar="SEED:MODEL")
    value.add_argument("--execute", action="store_true")
    value.add_argument("--resume-existing", action="store_true")
    value.add_argument("--format", choices=("json", "shell"), default="json")
    return value


def command(*parts) -> list[str]:
    return [str(part) for part in parts]


def parse_pairs(values: list[str], protocol: dict) -> set[tuple[int, str]]:
    allowed = {
        (int(seed), model["name"])
        for seed in protocol["seeds"]
        for model in protocol["models"]
        if model["train"]
    }
    pairs = set()
    for value in values:
        raw_seed, separator, model = value.partition(":")
        pair = (int(raw_seed), model) if separator and raw_seed.isdigit() else None
        if pair not in allowed:
            raise ValueError(f"unregistered Track G training pair: {value}")
        if pair in pairs:
            raise ValueError(f"duplicate --pair: {value}")
        pairs.add(pair)
    return pairs


def checkpoint_path(model: dict, seed: int) -> Path:
    return ROOT / model["checkpoint"].format(seed=seed)


def optional_data_args(data_dir: Path | None) -> list[str]:
    return [] if data_dir is None else ["--data-dir", str(data_dir)]


def preflight_jobs(protocol_path, protocol, data_dir, trained=False):
    job = command(
        sys.executable,
        ROOT / "src/semantic_delphi_ukb/preflight_track_g.py",
        "--protocol",
        protocol_path,
        *optional_data_args(data_dir),
    )
    if trained:
        job.append("--require-trained-baselines")
    return [job]


def training_jobs(protocol_path, protocol, device, data_dir, pairs=None, resume_existing=False):
    jobs = []
    for seed in protocol["seeds"]:
        for model in protocol["models"]:
            if not model["train"] or (pairs and (seed, model["name"]) not in pairs):
                continue
            run_dir = ROOT / protocol["output_root"] / "runs" / f"seed{seed}" / "pretraining" / model["name"]
            job = command(
                sys.executable,
                ROOT / "src/semantic_delphi_ukb/train_track_g_baseline.py",
                "--protocol",
                protocol_path,
                "--model",
                model["family"],
                "--seed",
                seed,
                "--run-dir",
                run_dir,
                "--device",
                device,
                *optional_data_args(data_dir),
            )
            if resume_existing:
                job.extend(["--resume", run_dir / "checkpoints/last.pt"])
            jobs.append(job)
    return jobs


def cohort_jobs(protocol_path, protocol, data_dir):
    return [command(
        sys.executable,
        ROOT / "src/semantic_delphi_ukb/build_track_g_cohort.py",
        "--protocol", protocol_path,
        "--split", "val",
        "--output", ROOT / protocol["cohort_manifests"]["validation"],
        *optional_data_args(data_dir),
    )]


def _slug(temperature, top_p, death_bias):
    return f"t{temperature:g}_p{top_p:g}_d{death_bias:g}".replace("-", "m").replace(".", "p")


def sampler_grid_jobs(protocol_path, protocol, device, data_dir):
    jobs = []
    selection = protocol["sampler_selection"]
    cohort = ROOT / protocol["cohort_manifests"]["validation"]
    for model in protocol["models"]:
        for seed in protocol["seeds"]:
            for temperature in selection["temperature_grid"]:
                for top_p in selection["top_p_grid"]:
                    for death_bias in selection["death_logit_bias_grid"]:
                        out = ROOT / protocol["output_root"] / "sampler_grid" / model["name"] / f"seed{seed}" / _slug(temperature, top_p, death_bias)
                        jobs.append(command(
                            sys.executable,
                            ROOT / "src/semantic_delphi_ukb/evaluate_track_g_generation.py",
                            "--protocol", protocol_path,
                            "--model", model["name"],
                            "--seed", seed,
                            "--checkpoint", checkpoint_path(model, seed),
                            "--split", "val",
                            "--out-dir", out,
                            "--device", device,
                            "--max-patients", selection["max_patients"],
                            "--num-rollouts", selection["num_rollouts"],
                            "--temperature", temperature,
                            "--top-p", top_p,
                            "--death-logit-bias", death_bias,
                            "--cohort-manifest", cohort,
                            *optional_data_args(data_dir),
                        ))
    return jobs


def sampler_select_jobs(protocol_path, protocol):
    root = ROOT / protocol["output_root"]
    return [command(
        sys.executable,
        ROOT / "src/semantic_delphi_ukb/select_track_g_sampler.py",
        "--protocol", protocol_path,
        "--grid-root", root / "sampler_grid",
        "--output", root / "manifests/sampler_selection.json",
    )]


def validation_jobs(protocol_path, protocol, device, data_dir):
    jobs = []
    root = ROOT / protocol["output_root"]
    manifest = root / "manifests/sampler_selection.json"
    cohort = ROOT / protocol["cohort_manifests"]["validation"]
    for seed in protocol["seeds"]:
        for model in protocol["models"]:
            jobs.append(command(
                sys.executable,
                ROOT / "src/semantic_delphi_ukb/evaluate_track_g_generation.py",
                "--protocol", protocol_path,
                "--model", model["name"],
                "--seed", seed,
                "--checkpoint", checkpoint_path(model, seed),
                "--split", "val",
                "--out-dir", root / "val" / f"seed{seed}" / model["name"],
                "--device", device,
                "--sampler-manifest", manifest,
                "--cohort-manifest", cohort,
                *optional_data_args(data_dir),
            ))
    return jobs


def finalize_jobs(protocol_path, protocol):
    root = ROOT / protocol["output_root"]
    return [command(
        sys.executable,
        ROOT / "src/semantic_delphi_ukb/finalize_track_g.py",
        "--protocol", protocol_path,
        "--split", "val",
        "--input-root", root / "val",
        "--out-dir", root / "val_assessment",
    )]


def execute_job(job: list[str]) -> None:
    if any(value.endswith("train_track_g_baseline.py") for value in job) and "--run-dir" in job:
        run_dir = Path(job[job.index("--run-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
            subprocess.run(job, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(job, cwd=ROOT, check=True)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_g_protocol(args.protocol, ROOT)
    if args.pair and args.lane != "train":
        raise ValueError("--pair is valid only with --lane train")
    pairs = parse_pairs(args.pair, protocol)
    lanes = {
        "preflight": preflight_jobs(args.protocol, protocol, args.data_dir),
        "cohort": cohort_jobs(args.protocol, protocol, args.data_dir),
        "train": training_jobs(
            args.protocol,
            protocol,
            args.device,
            args.data_dir,
            pairs or None,
            args.resume_existing,
        ),
        "sampler-grid": preflight_jobs(args.protocol, protocol, args.data_dir, trained=True)
        + sampler_grid_jobs(args.protocol, protocol, args.device, args.data_dir),
        "sampler-select": sampler_select_jobs(args.protocol, protocol),
        "validate": validation_jobs(args.protocol, protocol, args.device, args.data_dir),
        "finalize": finalize_jobs(args.protocol, protocol),
    }
    selected = list(lanes) if args.lane == "all" else [args.lane]
    rendered = {name: [shlex.join(job) for job in lanes[name]] for name in selected}
    if args.format == "shell":
        for name in selected:
            print(f"# lane: {name}")
            print("\n".join(rendered[name]))
    else:
        print(json.dumps({
            "execute": args.execute,
            "protocol_id": protocol["protocol_id"],
            "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
            "locked_test_read": protocol["locked_test_read"],
            "test_authorized": protocol["test_authorized"],
            "pairs": sorted(f"{seed}:{model}" for seed, model in pairs),
            "lanes": rendered,
            "test_commands_included": False,
        }, indent=2))
    if not args.execute:
        return 0
    if any(name not in {"preflight", "cohort"} for name in selected):
        assert_training_allowed(protocol)
    for name in selected:
        for job in lanes[name]:
            execute_job(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
