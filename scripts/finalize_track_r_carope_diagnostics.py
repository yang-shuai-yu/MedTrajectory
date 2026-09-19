"""Finalize Track R CARoPE diagnostics without mixing incompatible contracts."""

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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--run-tag", default="seed42_diagnostic2")
    p.add_argument("--hybrid-run-tag", default="seed42_hybridage1")
    p.add_argument("--current-run-tag", default="seed42_retry1")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    return p


def _validate_tag(value: str, option: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{option} must be a single non-empty directory name")
    return value


def _compare_command(inputs: dict[str, Path], out_dir: Path, bootstrap: int, seed: int, factorial=False):
    command = [sys.executable, "-m", "semantic_delphi_ukb.compare_horizon_control_tasks"]
    for name, path in inputs.items():
        command.extend(("--input", f"{name}={path}"))
    command.extend((
        "--out-dir", str(out_dir),
        "--bootstrap", str(bootstrap),
        "--seed", str(seed),
        "--memory-efficient",
    ))
    if factorial:
        command.extend((
            "--factorial-interaction",
            "A1-TokenStatic,A0-TokenStatic,A1-noStatic,A0-noStatic",
        ))
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_report(rebuild_summary_path: Path, historical_path: Path, output: Path) -> None:
    rebuild_payload = json.loads(rebuild_summary_path.read_text(encoding="utf-8-sig"))
    historical_payload = json.loads(historical_path.read_text(encoding="utf-8-sig"))
    rebuild = rebuild_payload["summary"]["macro"]
    historical = historical_payload["models"]["A1"]
    payload = {
        "comparison_role": "cross_contract_descriptive_rebuild_check",
        "paired_bootstrap": False,
        "reason": "source_data_dir and legacy evaluator are not paired with Track R output_data_dir rows",
        "historical_reference": {
            "path": str(historical_path.resolve()),
            "sha256": _sha256(historical_path),
            "model": "A1",
            "metrics": historical,
        },
        "legacy_contract_rebuild": {
            "path": str(rebuild_summary_path.resolve()),
            "sha256": _sha256(rebuild_summary_path),
            "metrics": rebuild,
        },
        "descriptive_delta_auc": float(rebuild["auc"] - historical["auc"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Legacy Contract Rebuild",
        "",
        "This is a descriptive comparison against the historical A1 validation result. It is not patient-paired and is excluded from the Track R 2x2 interaction.",
        "",
        "| Result | Macro AUC | Macro AUPRC | Macro Brier | Macro ECE |",
        "|---|---:|---:|---:|---:|",
        f"| Historical A1 | {historical['auc']:.6f} | {historical['auprc']:.6f} | {historical['brier']:.6f} | {historical['ece']:.6f} |",
        f"| Legacy contract rebuild | {rebuild['auc']:.6f} | {rebuild['auprc']:.6f} | {rebuild['brier']:.6f} | {rebuild['ece']:.6f} |",
        "",
        f"Descriptive delta AUC: `{payload['descriptive_delta_auc']:+.6f}`",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def jobs(args, protocol):
    run_tag = _validate_tag(args.run_tag, "--run-tag")
    hybrid_run_tag = _validate_tag(args.hybrid_run_tag, "--hybrid-run-tag")
    current_tag = _validate_tag(args.current_run_tag, "--current-run-tag")
    track_root = ROOT / protocol["output_root"]
    diagnostics = track_root / "diagnostics/carope_static_integration" / run_tag
    hybrid_diagnostics = track_root / "diagnostics/carope_static_integration" / hybrid_run_tag
    current = track_root / "runs" / current_tag
    analysis = diagnostics / "analysis"
    factorial = {
        "A0-TokenStatic": current / "A0-TokenStatic/validation/rows.json",
        "A0-noStatic": diagnostics / "A0-noStatic/validation/rows.json",
        "A1-TokenStatic": current / "A1-TokenStatic/validation/rows.json",
        "A1-noStatic": current / "A1-noStatic/validation/rows.json",
    }
    extended = {
        **factorial,
        "A1-LegacySex-BOS": diagnostics / "A1-LegacySex-BOS/validation/rows.json",
        "A1-DynamicRoPE-PostStatic": diagnostics / "A1-DynamicRoPE-PostStatic/validation/rows.json",
        "CARoPE-HybridAge-PostStatic": hybrid_diagnostics / "CARoPE-HybridAge-PostStatic/validation/rows.json",
    }
    drift = {
        "A1-TokenStatic-best": current / "A1-TokenStatic/validation/rows.json",
        "A1-TokenStatic-last": diagnostics / "A1-TokenStatic-last/validation/rows.json",
    }
    return {
        "analysis_root": analysis,
        "factorial": _compare_command(factorial, analysis / "factorial_2x2", args.bootstrap, args.seed, factorial=True),
        "extended": _compare_command(extended, analysis / "same_source_extended", args.bootstrap, args.seed),
        "checkpoint_drift": _compare_command(drift, analysis / "checkpoint_drift", args.bootstrap, args.seed),
        "legacy_rebuild_summary": diagnostics / "A1-LegacyContract-Rebuild/validation/summary.json",
        "legacy_historical": ROOT / protocol["carope_diagnostic_analysis"]["cross_contract_rebuild"]["historical_validation_reference"],
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    plan = jobs(args, protocol)
    rendered = {
        "execute": args.execute,
        "locked_test_read": False,
        "analysis_root": str(plan["analysis_root"]),
        "paired_commands": {
            name: shlex.join(plan[name])
            for name in ("factorial", "extended", "checkpoint_drift")
        },
        "cross_contract_descriptive": {
            "rebuild_summary": str(plan["legacy_rebuild_summary"]),
            "historical_reference": str(plan["legacy_historical"]),
            "paired_bootstrap": False,
        },
    }
    print(json.dumps(rendered, indent=2))
    if not args.execute:
        return 0
    if plan["analysis_root"].exists():
        raise FileExistsError(f"analysis output already exists: {plan['analysis_root']}")
    required = []
    for name in ("factorial", "extended", "checkpoint_drift"):
        required.extend(Path(item.split("=", 1)[1]) for item in plan[name] if "=" in item)
    required.extend((plan["legacy_rebuild_summary"], plan["legacy_historical"]))
    missing = sorted({str(path) for path in required if not path.is_file()})
    if missing:
        raise FileNotFoundError(f"diagnostic inputs are incomplete: {missing}")
    for name in ("factorial", "extended", "checkpoint_drift"):
        subprocess.run(plan[name], cwd=ROOT, check=True)
    _legacy_report(
        plan["legacy_rebuild_summary"],
        plan["legacy_historical"],
        plan["analysis_root"] / "legacy_contract_rebuild",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
