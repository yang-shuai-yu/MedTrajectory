"""Render or run the ordered validation-only CARoPE static-integration diagnostics."""

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


STAGE_ORDER = (
    "checkpoint_drift",
    "a0_no_static",
    "legacy_sex_bos",
    "post_trunk_static",
    "hybrid_age_poststatic",
    "legacy_contract_rebuild",
)

HYBRID_SPEC = ROOT / "configs/paper_protocol_v1/TRACK_R_hybrid_age_poststatic_validation_v1.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--hybrid-spec", type=Path, default=HYBRID_SPEC)
    p.add_argument("--stage", choices=(*STAGE_ORDER, "all"), default="all")
    p.add_argument("--run-tag", default="seed42_diagnostic1")
    p.add_argument("--current-run-tag", default="seed42_retry1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true")
    return p


def command(*parts: object) -> list[str]:
    return [str(part) for part in parts]


def _validate_tag(value: str, option: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{option} must be a single non-empty directory name")
    return value


def _training_common(
    data_dir: Path,
    diseases_yaml: Path,
    device: str,
    age_encoding: str = "carope",
) -> list[object]:
    if age_encoding not in ("carope", "sincos", "hybrid"):
        raise ValueError(f"unsupported age encoding: {age_encoding}")
    return [
        "--data-dir", data_dir,
        "--diseases-yaml", diseases_yaml,
        "--device", device,
        "--seed", "42",
        "--block-size", "128",
        "--n-layer", "6",
        "--n-head", "8",
        "--n-embd", "64",
        "--use-age-encoding", str(age_encoding in ("sincos", "hybrid")).lower(),
        "--use-age-rope", str(age_encoding in ("carope", "hybrid")).lower(),
        "--use-relative-horizon-query", "false",
        "--time-gap-loss-weight", "0.0",
        "--horizons", "1,5,10",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_hybrid_spec(args: argparse.Namespace) -> dict:
    spec = json.loads(args.hybrid_spec.read_text(encoding="utf-8"))
    if spec["parent_protocol_id"] != "track_r_v2_1":
        raise ValueError("hybrid addendum must reference track_r_v2_1")
    actual_parent_hash = _sha256(args.protocol)
    if spec["parent_protocol_sha256"] != actual_parent_hash:
        raise ValueError(
            "hybrid addendum parent protocol hash mismatch: "
            f"registered={spec['parent_protocol_sha256']} actual={actual_parent_hash}"
        )
    model = spec["model"]
    expected = {
        "use_age_encoding": True,
        "use_age_rope": True,
        "include_static_prefix": False,
        "static_conditioning": "categorical-residual",
        "static_fusion_stage": "post_transformer",
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"hybrid addendum requires model.{key}={value!r}")
    if spec.get("split") != "val" or spec.get("locked_test_read") is not False:
        raise ValueError("hybrid addendum must be validation-only and forbid locked-test reads")
    return spec


def _legacy_contract_rebuild_jobs(
    protocol: dict,
    val_landmarks: Path,
    run: Path,
    device: str,
) -> list[list[str]]:
    data = Path(protocol["source_data_dir"])
    common = _training_common(data, ROOT / protocol["diseases_yaml"], device)
    return [
        command(
            sys.executable, ROOT / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
            "--run-dir", run / "pretraining", "--static-fusion-stage", "pre_transformer", *common,
        ),
        command(
            sys.executable, ROOT / "src/semantic_delphi_ukb/train_car_rope.py",
            "--run-dir", run / "horizon",
            "--init-from-ckpt", run / "pretraining/checkpoints/best_val_loss.pt",
            "--static-fusion-stage", "pre_transformer", *common,
        ),
        command(
            sys.executable, "-m", "semantic_delphi_ukb.evaluate_medical_control_tasks",
            "--mode", "explicit", "--checkpoint", run / "horizon/checkpoints/best_val_horizon_auc.pt",
            "--data-dir", data, "--split", "val", "--diseases-yaml", ROOT / protocol["diseases_yaml"],
            "--out-dir", run / "validation", "--device", device, "--block-size", "128",
            "--horizons", "1,5,10", "--age-groups", "50,55,60,65,70,75",
            "--landmark-manifest", val_landmarks,
        ),
    ]


def _track_r_variant_jobs(
    protocol_path: Path,
    protocol: dict,
    val_landmarks: Path,
    run: Path,
    device: str,
    model_name: str,
    static_conditioning: str,
    fusion_stage: str,
    age_encoding: str = "carope",
) -> list[list[str]]:
    data = Path(protocol["output_data_dir"])
    common = [
        "--track-r-protocol", protocol_path,
        "--include-static-prefix", "false",
        "--static-conditioning", static_conditioning,
        "--static-fusion-stage", fusion_stage,
        *_training_common(
            data,
            ROOT / protocol["diseases_yaml"],
            device,
            age_encoding=age_encoding,
        ),
    ]
    return [
        command(
            sys.executable, ROOT / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
            "--run-dir", run / "pretraining", *common,
        ),
        command(
            sys.executable, ROOT / "src/semantic_delphi_ukb/train_car_rope.py",
            "--run-dir", run / "horizon",
            "--init-from-ckpt", run / "pretraining/checkpoints/best_val_loss.pt", *common,
        ),
        command(
            sys.executable, "-m", "semantic_delphi_ukb.evaluate_track_r",
            "--family", "carope", "--model-name", model_name,
            "--checkpoint", run / "horizon/checkpoints/best_val_horizon_auc.pt",
            "--protocol", protocol_path, "--data-dir", data, "--split", "val",
            "--landmark-manifest", val_landmarks, "--out-dir", run / "validation",
            "--include-static-prefix", "false", "--static-conditioning", static_conditioning,
            "--device", device,
        ),
    ]


def jobs(args: argparse.Namespace, protocol: dict) -> tuple[Path, dict[str, list[list[str]]]]:
    run_tag = _validate_tag(args.run_tag, "--run-tag")
    current_tag = _validate_tag(args.current_run_tag, "--current-run-tag")
    track_root = ROOT / protocol["output_root"]
    out_root = track_root / "diagnostics/carope_static_integration" / run_tag
    landmarks = track_root / "manifests/val_shared_landmarks.json"
    hybrid_spec = _load_hybrid_spec(args)
    hybrid_model = hybrid_spec["model"]
    stages = {
        "checkpoint_drift": [command(
            sys.executable, "-m", "semantic_delphi_ukb.evaluate_track_r",
            "--family", "carope", "--model-name", "A1-TokenStatic-last",
            "--checkpoint", track_root / "runs" / current_tag / "A1-TokenStatic/horizon/checkpoints/last.pt",
            "--protocol", args.protocol, "--data-dir", protocol["output_data_dir"], "--split", "val",
            "--landmark-manifest", landmarks, "--out-dir", out_root / "A1-TokenStatic-last/validation",
            "--include-static-prefix", "true", "--static-conditioning", "none", "--device", args.device,
        )],
        "a0_no_static": _track_r_variant_jobs(
            args.protocol, protocol, landmarks, out_root / "A0-noStatic", args.device,
            "A0-noStatic", "none", "pre_transformer", age_encoding="sincos",
        ),
        "legacy_sex_bos": _track_r_variant_jobs(
            args.protocol, protocol, landmarks, out_root / "A1-LegacySex-BOS", args.device,
            "A1-LegacySex-BOS", "legacy-sex-residual", "pre_transformer",
        ),
        "post_trunk_static": _track_r_variant_jobs(
            args.protocol, protocol, landmarks, out_root / "A1-DynamicRoPE-PostStatic", args.device,
            "A1-DynamicRoPE-PostStatic", "categorical-residual", "post_transformer",
        ),
        "hybrid_age_poststatic": _track_r_variant_jobs(
            args.protocol, protocol, landmarks, out_root / hybrid_model["name"], args.device,
            hybrid_model["name"], hybrid_model["static_conditioning"],
            hybrid_model["static_fusion_stage"], age_encoding="hybrid",
        ),
        "legacy_contract_rebuild": _legacy_contract_rebuild_jobs(
            protocol, landmarks, out_root / "A1-LegacyContract-Rebuild", args.device
        ),
    }
    return out_root, stages


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    hybrid_spec = _load_hybrid_spec(args)
    out_root, stages = jobs(args, protocol)
    selected = list(STAGE_ORDER) if args.stage == "all" else [args.stage]
    print(json.dumps({
        "execute": args.execute,
        "protocol": protocol["protocol_id"],
        "split": "val",
        "run_tag": args.run_tag,
        "output_root": str(out_root),
        "stage_order": list(STAGE_ORDER),
        "selected_stages": selected,
        "jobs": {name: [shlex.join(item) for item in stages[name]] for name in selected},
        "locked_test_read": False,
        "test_commands_included": False,
        "analysis_layers": protocol["carope_diagnostic_analysis"],
        "paired_bootstrap_scope": "output_data_dir plus evaluate_track_r rows only",
        "legacy_contract_rebuild_reporting": "descriptive comparison against the registered historical A1 validation result only",
        "post_trunk_static_contract": "categorical residual is gated by recruitment age and fused after all RoPE transformer blocks",
        "hybrid_age_poststatic_contract": hybrid_spec,
    }, indent=2))
    if not args.execute:
        return 0
    if protocol["status"] != "ready_for_build_after_static_field_audit":
        raise ValueError("Track R protocol is not released for training")
    if args.stage == "all" and out_root.exists():
        raise FileExistsError(f"diagnostic output root already exists: {out_root}")
    for name in selected:
        if name == "hybrid_age_poststatic":
            model_root = out_root / _load_hybrid_spec(args)["model"]["name"]
            model_root.mkdir(parents=True, exist_ok=False)
            manifest = {
                "spec_path": str(args.hybrid_spec.resolve()),
                "spec_sha256": _sha256(args.hybrid_spec),
                "parent_protocol_path": str(args.protocol.resolve()),
                "parent_protocol_sha256": _sha256(args.protocol),
                "locked_test_read": False,
                "commands": [shlex.join(item) for item in stages[name]],
            }
            (model_root / "experiment_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        for item in stages[name]:
            subprocess.run(item, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
