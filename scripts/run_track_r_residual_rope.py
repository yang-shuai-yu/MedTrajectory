"""Render or run the final validation-only static-safe residual-RoPE experiment."""

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

from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402


SPEC = ROOT / "configs/paper_protocol_v1/TRACK_R_static_safe_residual_rope_validation_v1.json"
STAGES = ("learned", "fixed1")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--spec", type=Path, default=SPEC)
    p.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    p.add_argument("--run-tag", default="seed42_residualrope1")
    p.add_argument("--a0-run-tag", default="seed42_retry1")
    p.add_argument("--a0-checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_tag(value: str, option: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{option} must be a single non-empty directory name")
    return value


def command(*parts: object) -> list[str]:
    return [str(part) for part in parts]


def load_spec(args: argparse.Namespace) -> dict:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("parent_protocol_id") != "track_r_v2_1":
        raise ValueError("residual-RoPE addendum must reference track_r_v2_1")
    actual = sha256(args.protocol)
    if spec.get("parent_protocol_sha256") != actual:
        raise ValueError(
            "residual-RoPE addendum parent hash mismatch: "
            f"registered={spec.get('parent_protocol_sha256')} actual={actual}"
        )
    architecture = spec["architecture"]
    required = {
        "base": "A0-TokenStatic",
        "base_parameters_initialized_from_existing_a0_and_frozen": True,
        "residual_scope": "clinical_dynamic_query_to_clinical_dynamic_key_only",
        "alpha_initial": 0.01,
        "inner_rope_gate": False,
        "phase_origin": "recruitment_age",
    }
    for key, value in required.items():
        if architecture.get(key) != value:
            raise ValueError(f"residual-RoPE addendum requires architecture.{key}={value!r}")
    if spec.get("split") != "val" or spec.get("locked_test_read") is not False:
        raise ValueError("residual-RoPE addendum must be validation-only and forbid locked-test reads")
    return spec


def training_command(
    args: argparse.Namespace,
    protocol: dict,
    run_dir: Path,
    a0_checkpoint: Path,
    mode: str,
) -> list[str]:
    fixed_alpha = "1.0" if mode == "fixed" else "1.0"
    return command(
        sys.executable,
        ROOT / "src/semantic_delphi_ukb/train_car_rope.py",
        "--data-dir", protocol["output_data_dir"],
        "--diseases-yaml", ROOT / protocol["diseases_yaml"],
        "--run-dir", run_dir,
        "--init-from-a0-ckpt", a0_checkpoint,
        "--track-r-protocol", args.protocol,
        "--include-static-prefix", "true",
        "--static-conditioning", "none",
        "--static-fusion-stage", "pre_transformer",
        "--device", args.device,
        "--seed", "42",
        "--block-size", "128",
        "--n-layer", "6",
        "--n-head", "8",
        "--n-embd", "64",
        "--use-age-encoding", "true",
        "--use-age-rope", "false",
        "--use-relative-horizon-query", "false",
        "--time-gap-loss-weight", "0.0",
        "--horizons", "1,5,10",
        "--rope-scales", "0.25,0.75,2,4",
        "--residual-rope-mode", mode,
        "--residual-rope-initial-alpha", "0.01",
        "--residual-rope-fixed-alpha", fixed_alpha,
        "--residual-rope-phase-origin", "recruitment_age",
    )


def evaluation_command(
    args: argparse.Namespace,
    protocol: dict,
    landmarks: Path,
    checkpoint: Path,
    model_name: str,
    output: Path,
    alpha_override: float | None = None,
) -> list[str]:
    parts: list[object] = [
        sys.executable,
        "-m",
        "semantic_delphi_ukb.evaluate_track_r",
        "--family", "carope",
        "--model-name", model_name,
        "--checkpoint", checkpoint,
        "--protocol", args.protocol,
        "--data-dir", protocol["output_data_dir"],
        "--split", "val",
        "--landmark-manifest", landmarks,
        "--out-dir", output,
        "--include-static-prefix", "true",
        "--static-conditioning", "none",
        "--device", args.device,
    ]
    if alpha_override is not None:
        parts.extend(("--residual-rope-alpha-override", str(alpha_override)))
    return command(*parts)


def jobs(args: argparse.Namespace, protocol: dict) -> tuple[Path, Path, dict[str, list[list[str]]]]:
    run_tag = validate_tag(args.run_tag, "--run-tag")
    a0_tag = validate_tag(args.a0_run_tag, "--a0-run-tag")
    track_root = ROOT / protocol["output_root"]
    output_root = track_root / "diagnostics/static_safe_residual_rope" / run_tag
    a0_checkpoint = args.a0_checkpoint or (
        track_root / "runs" / a0_tag / "A0-TokenStatic/horizon/checkpoints/best_val_horizon_auc.pt"
    )
    landmarks = track_root / "manifests/val_shared_landmarks.json"
    learned_root = output_root / "A1-ResidualRoPE-Learned"
    fixed_root = output_root / "A1-ResidualRoPE-Fixed1"
    learned_checkpoint = learned_root / "horizon/checkpoints/best_val_horizon_auc.pt"
    fixed_checkpoint = fixed_root / "horizon/checkpoints/best_val_horizon_auc.pt"
    stages = {
        "learned": [
            training_command(args, protocol, learned_root / "horizon", a0_checkpoint, "learned"),
            evaluation_command(
                args, protocol, landmarks, learned_checkpoint,
                "A1-ResidualRoPE-Learned", learned_root / "validation",
            ),
            evaluation_command(
                args, protocol, landmarks, learned_checkpoint,
                "A1-ResidualRoPE-Learned-Alpha0", learned_root / "validation_alpha0", 0.0,
            ),
            command(
                sys.executable,
                ROOT / "scripts/report_residual_rope_checkpoint.py",
                "--checkpoint", learned_checkpoint,
                "--output", learned_root / "residual_rope_checkpoint_report.json",
            ),
        ],
        "fixed1": [
            training_command(args, protocol, fixed_root / "horizon", a0_checkpoint, "fixed"),
            evaluation_command(
                args, protocol, landmarks, fixed_checkpoint,
                "A1-ResidualRoPE-Fixed1", fixed_root / "validation",
            ),
            command(
                sys.executable,
                ROOT / "scripts/report_residual_rope_checkpoint.py",
                "--checkpoint", fixed_checkpoint,
                "--output", fixed_root / "residual_rope_checkpoint_report.json",
            ),
        ],
    }
    return output_root, a0_checkpoint, stages


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    spec = load_spec(args)
    output_root, a0_checkpoint, stages = jobs(args, protocol)
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    print(json.dumps({
        "execute": args.execute,
        "protocol": protocol["protocol_id"],
        "spec": str(args.spec),
        "spec_sha256": sha256(args.spec),
        "split": "val",
        "run_tag": args.run_tag,
        "output_root": str(output_root),
        "frozen_a0_checkpoint": str(a0_checkpoint),
        "selected_stages": selected,
        "jobs": {name: [shlex.join(item) for item in stages[name]] for name in selected},
        "fallback_target": spec["fallback_target"],
        "locked_test_read": False,
        "test_commands_included": False,
    }, indent=2))
    if not args.execute:
        return 0
    if protocol["status"] != "ready_for_build_after_static_field_audit":
        raise ValueError("Track R protocol is not released for training")
    if not a0_checkpoint.is_file():
        raise FileNotFoundError(f"frozen A0 checkpoint was not found: {a0_checkpoint}")
    if not (ROOT / protocol["output_root"] / "manifests/val_shared_landmarks.json").is_file():
        raise FileNotFoundError("frozen validation landmark manifest was not found")
    for stage in selected:
        model_root = output_root / (
            "A1-ResidualRoPE-Learned" if stage == "learned" else "A1-ResidualRoPE-Fixed1"
        )
        model_root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "spec_path": str(args.spec.resolve()),
            "spec_sha256": sha256(args.spec),
            "parent_protocol_path": str(args.protocol.resolve()),
            "parent_protocol_sha256": sha256(args.protocol),
            "frozen_a0_checkpoint": str(a0_checkpoint.resolve()),
            "frozen_a0_checkpoint_sha256": sha256(a0_checkpoint),
            "stage": stage,
            "locked_test_read": False,
            "commands": [shlex.join(item) for item in stages[stage]],
        }
        (model_root / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for item in stages[stage]:
            subprocess.run(item, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
