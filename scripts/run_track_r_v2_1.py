"""Validate and render Track R v2.1 commands; execute only with an explicit flag."""
from __future__ import annotations

try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import EXTERNAL_ROOT

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--lane", choices=("audit", "build", "transformers", "baselines", "medbert", "all"), default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--run-tag",
        default="seed42",
        help="immutable run-directory name under results/track_r_v2_1/runs",
    )
    p.add_argument("--execute", action="store_true")
    return p


def command(*parts) -> list[str]:
    return [str(part) for part in parts]


def paths(protocol: dict, run_tag: str = "seed42") -> dict[str, Path]:
    if not run_tag or Path(run_tag).name != run_tag:
        raise ValueError("--run-tag must be a single non-empty directory name")
    output_root = ROOT / protocol["output_root"]
    return {
        "data": Path(protocol["output_data_dir"]),
        "audit": output_root / "manifests/static_field_audit.json",
        "capacity": output_root / "manifests/model_capacity.json",
        "train_landmarks": output_root / "manifests/train_shared_landmarks.json",
        "val_landmarks": output_root / "manifests/val_shared_landmarks.json",
        "test_landmarks": output_root / "manifests/test_shared_landmarks.json",
        "features": output_root / "landmark_features",
        "runs": output_root / "runs" / run_tag,
        "validation_freeze": output_root / "validation_freeze_manifest.json",
    }


def audit_jobs(protocol_path: Path, protocol: dict, p: dict[str, Path]) -> list[list[str]]:
    return [command(
        sys.executable, ROOT / "scripts/audit_track_r_static_fields.py",
        "--protocol", protocol_path,
        "--ukb-extract-dir", str(EXTERNAL_ROOT),
        "--output", p["audit"],
    )]


def build_jobs(protocol_path: Path, protocol: dict, p: dict[str, Path]) -> list[list[str]]:
    source = Path(protocol["source_data_dir"])
    jobs = [
        command(
            sys.executable, ROOT / "scripts/build_track_r_static_overlay.py",
            "--protocol", protocol_path, "--audit", p["audit"],
            "--ukb-extract-dir", str(EXTERNAL_ROOT),
            "--source-data-dir", source, "--output-dir", p["data"],
        ),
        command(
            sys.executable, ROOT / "scripts/report_track_r_model_capacity.py",
            "--protocol", protocol_path, "--data-dir", p["data"], "--output", p["capacity"],
        ),
    ]
    for split in ("train", "val"):
        jobs.append(command(
            sys.executable, ROOT / "scripts/build_shared_test_landmarks.py",
            "--data-dir", p["data"], "--split", split,
            "--age-groups", "50,55,60,65,70,75", "--seed", "1337", "--block-size", "128",
            "--out", p[f"{split}_landmarks"],
        ))
        jobs.append(command(
            sys.executable, ROOT / "scripts/build_track_r_landmark_features.py",
            "--protocol", protocol_path, "--data-dir", p["data"], "--split", split,
            "--landmark-manifest", p[f"{split}_landmarks"],
            "--output", p["features"] / f"{split}.npz",
        ))
    return jobs


def transformer_jobs(protocol_path: Path, protocol: dict, p: dict[str, Path], device: str) -> list[list[str]]:
    jobs = []
    for variant in protocol["transformer_variants"]:
        run = p["runs"] / variant["name"]
        common = [
            "--track-r-protocol", protocol_path,
            "--data-dir", p["data"], "--diseases-yaml", ROOT / protocol["diseases_yaml"],
            "--device", device, "--seed", "42", "--block-size", str(protocol["dynamic_context_length"]),
            "--n-layer", str(variant["n_layer"]), "--n-head", str(variant["n_head"]), "--n-embd", str(variant["n_embd"]),
            "--include-static-prefix", str(variant["static_prefix"]).lower(),
            "--use-age-encoding", str(variant["age_encoding"] == "sincos").lower(),
            "--use-age-rope", str(variant["age_encoding"] == "carope").lower(),
            "--use-relative-horizon-query", "false", "--time-gap-loss-weight", "0.0",
            "--horizons", "1,5,10",
        ]
        jobs.append(command(sys.executable, ROOT / "src/semantic_delphi_ukb/train_car_rope_pretraining.py", "--run-dir", run / "pretraining", *common))
        jobs.append(command(
            sys.executable, ROOT / "src/semantic_delphi_ukb/train_car_rope.py", "--run-dir", run / "horizon",
            "--init-from-ckpt", run / "pretraining/checkpoints/best_val_loss.pt", *common,
        ))
        jobs.append(command(
            sys.executable, "-m", "semantic_delphi_ukb.evaluate_track_r",
            "--family", "carope", "--model-name", variant["name"],
            "--checkpoint", run / "horizon/checkpoints/best_val_horizon_auc.pt",
            "--protocol", protocol_path, "--data-dir", p["data"], "--split", "val",
            "--landmark-manifest", p["val_landmarks"], "--out-dir", run / "validation",
            "--include-static-prefix", str(variant["static_prefix"]).lower(), "--device", device,
        ))
    return jobs


def baseline_jobs(protocol_path: Path, p: dict[str, Path]) -> list[list[str]]:
    jobs = []
    registered = (
        ("logistic", "Logistic-R"),
        ("cox", "Cox-R"),
        ("mdrmf-clinical", "MDRMF-Clinical-R"),
    )
    for name, registered_name in registered:
        jobs.append(command(
            sys.executable, "-m", "semantic_delphi_ukb.track_r_baselines",
            "--model", name,
            "--train-features", p["features"] / "train.npz",
            "--val-features", p["features"] / "val.npz",
            "--eval-features", p["features"] / "val.npz",
            "--output-dir", p["runs"] / registered_name / "validation",
        ))
    return jobs


def medbert_jobs(protocol_path: Path, protocol: dict, p: dict[str, Path], device: str) -> list[list[str]]:
    jobs = []
    for item in protocol["medbert_variants"]:
        run = p["runs"] / item["name"]
        common = ["--variant", item["name"], "--protocol", protocol_path, "--data-dir", p["data"], "--device", device, "--seed", "42"]
        jobs.append(command(sys.executable, "-m", "semantic_delphi_ukb.train_medbert_r", "--stage", "pretraining", "--run-dir", run / "pretraining", *common))
        jobs.append(command(
            sys.executable, "-m", "semantic_delphi_ukb.train_medbert_r", "--stage", "horizon", "--run-dir", run / "horizon",
            "--init-from-ckpt", run / "pretraining/checkpoints/best_val_loss.pt", *common,
        ))
        jobs.append(command(
            sys.executable, "-m", "semantic_delphi_ukb.evaluate_track_r",
            "--family", "medbert", "--model-name", item["name"],
            "--checkpoint", run / "horizon/checkpoints/best_val_horizon_auc.pt",
            "--protocol", protocol_path, "--data-dir", p["data"], "--split", "val",
            "--landmark-manifest", p["val_landmarks"], "--out-dir", run / "validation",
            "--include-static-prefix", "true", "--device", device,
        ))
    return jobs


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    p = paths(protocol, args.run_tag)
    lanes = {
        "audit": audit_jobs(args.protocol, protocol, p),
        "build": build_jobs(args.protocol, protocol, p),
        "transformers": transformer_jobs(args.protocol, protocol, p, args.device),
        "baselines": baseline_jobs(args.protocol, p),
        "medbert": medbert_jobs(args.protocol, protocol, p, args.device),
    }
    selected = list(lanes) if args.lane == "all" else [args.lane]
    rendered = {name: [shlex.join(job) for job in lanes[name]] for name in selected}
    print(json.dumps({
        "execute": args.execute,
        "run_tag": args.run_tag,
        "protocol_status": protocol["status"],
        "lanes": rendered,
        "parallel_after_build": ["transformers", "baselines", "medbert"],
        "test_commands_included": False,
        "test_gate": str(p["validation_freeze"]),
    }, indent=2))
    if not args.execute:
        return 0
    if protocol["status"] != "ready_for_build_after_static_field_audit" and any(name != "audit" for name in selected):
        raise ValueError("only the audit lane may execute while the protocol is still draft")
    for name in selected:
        for job in lanes[name]:
            subprocess.run(job, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
