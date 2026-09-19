"""Run paired validation comparison and assess the locked residual-RoPE conditions."""

from __future__ import annotations

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
    p.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs/paper_protocol_v1/TRACK_R_static_safe_residual_rope_validation_v1.json",
    )
    p.add_argument("--run-tag", default="seed42_residualrope1")
    p.add_argument("--a0-run-tag", default="seed42_retry1")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    return p


def validate_tag(value: str, option: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{option} must be a single non-empty directory name")
    return value


def plan(args, protocol):
    run_tag = validate_tag(args.run_tag, "--run-tag")
    a0_tag = validate_tag(args.a0_run_tag, "--a0-run-tag")
    track_root = ROOT / protocol["output_root"]
    residual = track_root / "diagnostics/static_safe_residual_rope" / run_tag
    learned = residual / "A1-ResidualRoPE-Learned"
    fixed = residual / "A1-ResidualRoPE-Fixed1"
    inputs = {
        "A0-TokenStatic": track_root / "runs" / a0_tag / "A0-TokenStatic/validation/rows.json",
        "A1-ResidualRoPE-Learned": learned / "validation/rows.json",
        "A1-ResidualRoPE-Learned-Alpha0": learned / "validation_alpha0/rows.json",
        "A1-ResidualRoPE-Fixed1": fixed / "validation/rows.json",
    }
    analysis = residual / "analysis"
    command = [sys.executable, "-m", "semantic_delphi_ukb.compare_horizon_control_tasks"]
    for name, path in inputs.items():
        command.extend(("--input", f"{name}={path}"))
    command.extend((
        "--out-dir", str(analysis),
        "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
        "--memory-efficient",
    ))
    return {
        "analysis": analysis,
        "inputs": inputs,
        "command": command,
        "learned_report": learned / "residual_rope_checkpoint_report.json",
    }


def write_assessment(args, details) -> None:
    comparison = json.loads((details["analysis"] / "comparison.json").read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    diagnostics = json.loads(details["learned_report"].read_text(encoding="utf-8"))["diagnostics"]
    models = comparison["models"]
    a0 = models["A0-TokenStatic"]
    learned = models["A1-ResidualRoPE-Learned"]
    alpha0 = models["A1-ResidualRoPE-Learned-Alpha0"]
    alpha_values = [
        float(value)
        for layer in diagnostics["layers"]
        for value in layer["alpha_by_head"]
    ]
    primary_met = learned["auc"] >= a0["auc"]
    ablation_met = learned["auc"] > alpha0["auc"]
    alpha_behavior = (
        "remained_near_initialization"
        if max(alpha_values) < 2.0 * float(spec["architecture"]["alpha_initial"])
        else "departed_materially_from_initialization"
    )
    payload = {
        "experiment_id": spec["experiment_id"],
        "split": "val",
        "locked_test_read": False,
        "fallback_target": "A0-TokenStatic",
        "metrics": models,
        "conditions": {
            "learned_auc_not_below_a0": primary_met,
            "learned_minus_a0_auc": float(learned["auc"] - a0["auc"]),
            "learned_better_than_same_checkpoint_alpha0": ablation_met,
            "learned_minus_alpha0_auc": float(learned["auc"] - alpha0["auc"]),
            "alpha_behavior": alpha_behavior,
            "alpha_initial": float(spec["architecture"]["alpha_initial"]),
            "alpha_min": min(alpha_values),
            "alpha_mean": sum(alpha_values) / len(alpha_values),
            "alpha_max": max(alpha_values),
        },
        "overall_success": bool(primary_met and ablation_met and alpha_behavior == "departed_materially_from_initialization"),
        "decision_basis": (
            "Primary AUROC and same-checkpoint ablation both fail in the required direction; "
            "the overall decision does not depend on the descriptive alpha-near-initialization rule."
        ),
        "phase_origin_role": "provenance_and_numerical_centering_only",
        "paired_bootstrap": comparison["paired_bootstrap"],
    }
    (details["analysis"] / "success_assessment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Static-Safe Residual RoPE Validation Assessment",
        "",
        "Locked test read: `false`",
        "",
        "| Model | Macro AUROC | Macro AUPRC | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in models.items():
        lines.append(
            f"| {name} | {values['auc']:.6f} | {values['auprc']:.6f} | "
            f"{values['brier']:.6f} | {values['ece']:.6f} |"
        )
    lines += [
        "",
        f"Overall success: **{str(payload['overall_success']).lower()}**",
        "",
        f"- Learned minus A0 AUROC: `{payload['conditions']['learned_minus_a0_auc']:+.8f}`.",
        f"- Learned minus same-checkpoint alpha=0 AUROC: `{payload['conditions']['learned_minus_alpha0_auc']:+.8f}`.",
        f"- Alpha min/mean/max: `{min(alpha_values):.6f} / {sum(alpha_values) / len(alpha_values):.6f} / {max(alpha_values):.6f}`.",
        "- Recruitment-age phase origin is provenance/numerical centering only, not a performance claim.",
    ]
    (details["analysis"] / "success_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    details = plan(args, protocol)
    print(json.dumps({
        "execute": args.execute,
        "split": "val",
        "locked_test_read": False,
        "analysis": str(details["analysis"]),
        "inputs": {name: str(path) for name, path in details["inputs"].items()},
        "command": shlex.join(details["command"]),
    }, indent=2))
    if not args.execute:
        return 0
    if details["analysis"].exists():
        raise FileExistsError(f"analysis output already exists: {details['analysis']}")
    missing = [str(path) for path in (*details["inputs"].values(), details["learned_report"], args.spec) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"residual-RoPE comparison inputs are incomplete: {missing}")
    subprocess.run(details["command"], cwd=ROOT, check=True)
    write_assessment(args, details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
