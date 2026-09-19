#!/usr/bin/env python3
"""Summarize the completed P0-P4 x seed42/43/44 paper-protocol matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scipy import stats


METRICS = {
    "lm_auc": "LM medical AUC",
    "risk_auc": "Horizon-risk AUC",
    "long_auc": "Longitudinal AUC",
}
PAIRS = [("P1", "P0"), ("P3", "P2"), ("P2", "P0"), ("P3", "P0"), ("P4", "P3")]
PAPER_MEAN = 0.7565
PAPER_MEDIAN = 0.7649


def summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    half = tcrit * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def paired(a: list[float], b: list[float]) -> dict[str, float]:
    d = [x - y for x, y in zip(a, b)]
    out = summarize(d)
    if len(d) > 1 and out["sd"] > 0:
        tstat, pvalue = stats.ttest_1samp(d, 0.0)
    else:
        tstat, pvalue = float("nan"), float("nan")
    out.update({"t": float(tstat), "p_two_sided": float(pvalue), "deltas": d})
    return out


def load_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    rows: dict[str, dict[int, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            exp = row["experiment"]
            seed = int(row["seed"])
            rows.setdefault(exp, {})[seed] = {
                "lm_auc": float(row["lm_auc"]),
                "risk_auc": float(row["risk_auc"]),
                "long_auc": float(row["long_auc"]),
            }
    expected = {42, 43, 44}
    for exp in ["P0", "P1", "P2", "P3", "P4"]:
        if set(rows.get(exp, {})) != expected:
            raise ValueError(f"{exp} does not have exactly seeds 42/43/44")
    return rows


def load_rows_from_stream(handle) -> dict[str, dict[int, dict[str, float]]]:
    rows: dict[str, dict[int, dict[str, float]]] = {}
    for row in csv.DictReader(handle, delimiter="\t"):
        exp = row["experiment"]
        seed = int(row["seed"])
        rows.setdefault(exp, {})[seed] = {
            "lm_auc": float(row["lm_auc"]),
            "risk_auc": float(row["risk_auc"]),
            "long_auc": float(row["long_auc"]),
        }
    expected = {42, 43, 44}
    for exp in ["P0", "P1", "P2", "P3", "P4"]:
        if set(rows.get(exp, {})) != expected:
            raise ValueError(f"{exp} does not have exactly seeds 42/43/44")
    return rows


def f(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6f}"


def make_report(rows: dict[str, dict[int, dict[str, float]]]) -> tuple[str, dict]:
    summary: dict = {"per_experiment": {}, "paired": {}, "paper_reference": {}}
    lines = [
        "# Paper Protocol Matrix: Formal Comparison (2026-08-07)",
        "",
        "Source: completed remote matrix `matrix_seeds42-44_my3_20260806_0252`; all values are validation summaries.",
        "",
        "## 1. Per-seed results",
        "",
        "| Experiment | Seed | LM medical AUC | Horizon-risk AUC | Longitudinal AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp in ["P0", "P1", "P2", "P3", "P4"]:
        summary["per_experiment"][exp] = {}
        for seed in [42, 43, 44]:
            vals = rows[exp][seed]
            summary["per_experiment"][exp][str(seed)] = vals
            lines.append(f"| {exp} | {seed} | {vals['lm_auc']:.6f} | {vals['risk_auc']:.6f} | {vals['long_auc']:.6f} |")
    lines += ["", "| Experiment | LM medical AUC mean +/- SD | Horizon-risk AUC mean +/- SD | Longitudinal AUC mean +/- SD |", "|---|---:|---:|---:|"]
    for exp in ["P0", "P1", "P2", "P3", "P4"]:
        summary["per_experiment"].setdefault(exp, {})
        stats_by_metric = {}
        for metric in METRICS:
            stats_by_metric[metric] = summarize([rows[exp][s][metric] for s in [42, 43, 44]])
        summary["per_experiment"][exp]["aggregate"] = stats_by_metric
        lines.append("| {} | {} +/- {} | {} +/- {} | {} +/- {} |".format(
            exp,
            f(stats_by_metric["lm_auc"]["mean"]), f(stats_by_metric["lm_auc"]["sd"]),
            f(stats_by_metric["risk_auc"]["mean"]), f(stats_by_metric["risk_auc"]["sd"]),
            f(stats_by_metric["long_auc"]["mean"]), f(stats_by_metric["long_auc"]["sd"]),
        ))
    lines += [
        "",
        "## 2. Paired seed comparisons",
        "",
        "Differences are `first experiment - second experiment`, paired by seed. The interval is a two-sided paired t 95% CI (n=3), so it is descriptive and not a high-powered significance claim.",
        "",
        "| Contrast | Metric | Seed deltas | Mean delta | 95% CI | p (two-sided) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for first, second in PAIRS:
        for metric, label in METRICS.items():
            result = paired(
                [rows[first][s][metric] for s in [42, 43, 44]],
                [rows[second][s][metric] for s in [42, 43, 44]],
            )
            key = f"{first}-{second}:{metric}"
            summary["paired"][key] = result
            ci = f"[{f(result['ci95_low'])}, {f(result['ci95_high'])}]"
            deltas = ", ".join(f"{x:+.6f}" for x in result["deltas"])
            lines.append(f"| {first} - {second} | {label} | {deltas} | {result['mean']:+.6f} | {ci} | {result['p_two_sided']:.4f} |")
    lines += [
        "",
        "## 3. Delphi-2M reference comparison",
        "",
        f"The paper notebook reference is mean AUC `{PAPER_MEAN:.4f}` and median AUC `{PAPER_MEDIAN:.4f}`. The following one-sample intervals compare our three LM-AUC seeds with the mean reference; this is directional because disease set, cohort and temporal protocol are not yet identical.",
        "",
        "| Experiment | Our LM AUC mean | Difference vs 0.7565 | 95% CI for difference |",
        "|---|---:|---:|---:|",
    ]
    for exp in ["P0", "P1", "P2", "P3", "P4"]:
        values = [rows[exp][s]["lm_auc"] for s in [42, 43, 44]]
        result = summarize([x - PAPER_MEAN for x in values])
        summary["paper_reference"][exp] = result
        lines.append(f"| {exp} | {sum(values)/3:.6f} | {result['mean']:+.6f} | [{f(result['ci95_low'])}, {f(result['ci95_high'])}] |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- The largest within-matrix effect is diagnosis+death to full multitype (P0 to P2/P3), not RoPE.",
        "- RoPE is a controlled ablation only within P0/P1 and P2/P3; P4 also changes the cohort and therefore is not a pure RoPE comparison.",
        "- Only LM medical AUC is the direct analogue of the paper's next-token AUC. Horizon-risk and longitudinal AUC answer different questions.",
        "- The numerical comparison with 0.7565 is not a claim of replication or superiority until disease-set and temporal-cohort alignment is completed.",
    ]
    return "\n".join(lines) + "\n", summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows_from_stream(__import__("sys").stdin) if str(args.input_tsv) == "-" else load_rows(args.input_tsv)
    report, payload = make_report(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
