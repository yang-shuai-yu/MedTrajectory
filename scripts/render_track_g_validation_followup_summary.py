"""Render concise Markdown tables for the Track G validation follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("A0", "A2", "ETHOS-Matched", "Foresight-Matched")
HORIZONS = ("1y", "5y", "10y")


def f(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hazard-assessment",
        type=Path,
        default=ROOT / "results/track_g_v1/val_death_hazard_v2_followup/assessment.json",
    )
    parser.add_argument(
        "--no-death-assessment",
        type=Path,
        default=ROOT / "results/track_g_v1/val_no_death_token/paired_assessment.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/track_g_v1/VALIDATION_FOLLOWUP_SUMMARY.md",
    )
    args = parser.parse_args()
    hazard = json.loads(args.hazard_assessment.read_text(encoding="utf-8"))
    no_death = json.loads(args.no_death_assessment.read_text(encoding="utf-8"))

    lines = [
        "# Track G validation follow-up summary",
        "",
        "## Independent censor-aware death head",
        "",
        "| Model | Horizon | N / cases | Prevalence | Mean p | AUROC | AUPRC | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for horizon in HORIZONS:
            row = hazard["models"][model]["calibrated"][horizon]
            lines.append(
                f"| {model} | {horizon} | {row['n']} / {row['cases']} | {f(row['prevalence'])} | "
                f"{f(row['mean_probability'])} | {f(row['auroc'])} | {f(row['auprc'])} | "
                f"{f(row['brier'])} | {f(row['ece'])} |"
            )

    lines.extend([
        "",
        "## Independent 10y head versus rollout on the same censor-valid patients",
        "",
        "| Model | N | Rollout Brier | Head Brier | Delta head-rollout [95% CI] | Rollout AUROC | Head AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        row = hazard["rollout_10y_comparison"][model]
        delta = row["head_minus_rollout"]
        ci = delta["delta_brier_ci95"]
        lines.append(
            f"| {model} | {row['rollout']['n']} | {f(row['rollout']['brier'])} | "
            f"{f(row['independent_head_calibrated']['brier'])} | {f(delta['delta_brier'])} "
            f"[{f(ci[0])}, {f(ci[1])}] | {f(row['rollout']['auroc'])} | "
            f"{f(row['independent_head_calibrated']['auroc'])} |"
        )

    lines.extend([
        "",
        "## No-death-token paired trajectory changes",
        "",
        "| Model | Metric | Raw | No-death | Delta [95% CI] | N |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in ("A0", "A2"):
        for metric, row in no_death["models"][model].items():
            lines.append(
                f"| {model} | {metric} | {f(row['raw_mean'])} | {f(row['no_death_mean'])} | "
                f"{f(row['delta_no_death_minus_raw'])} [{f(row['ci95_low'])}, {f(row['ci95_high'])}] | "
                f"{row['patient_count']} |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
