#!/usr/bin/env python3
"""Create compact human-facing tables and a primary locked-test figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str) -> str:
    if value in (None, "", "None"):
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return f"{number:.6f}"


def write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: List[Dict[str, str]], fields: List[str]) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            cells.append(value if field in {"records", "primary", "diagnostic_or_exploratory"} else fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def make_figure(rows: List[Dict[str, str]], output: Path) -> bool:
    models = ["P0_seed42", "P2_seed42", "P3_seed42"]
    metrics = [("auc", "AUC"), ("auprc", "AUPRC")]
    lookup = {(row["model"], row["metric"]): float(row["value"])
              for row in rows if row["task"] == "locked_test_horizon_risk"}
    colors = {"P0_seed42": "#35608d", "P2_seed42": "#2e8b72", "P3_seed42": "#b56b35"}
    width, height = 760, 390
    plot_top, plot_bottom = 58, 315
    panel_width, panel_gap = 330, 48
    max_value = 0.15
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Locked test horizon head AUC and AUPRC">',
           '<title>Locked test horizon-head performance</title>',
           '<desc>Macro AUC and AUPRC for P0, P2 and P3 on the shared exact landmark locked test.</desc>',
           '<rect width="100%" height="100%" fill="#fbfaf7"/>',
           '<text x="380" y="28" text-anchor="middle" font-family="Arial" font-size="16" fill="#20252b">Locked test horizon-head performance</text>']
    for panel, (metric, label) in enumerate(metrics):
        left = 58 + panel * (panel_width + panel_gap)
        svg.extend([f'<text x="{left + panel_width / 2}" y="50" text-anchor="middle" font-family="Arial" font-size="14" fill="#20252b">{label}</text>',
                    f'<line x1="{left}" y1="{plot_bottom}" x2="{left + panel_width}" y2="{plot_bottom}" stroke="#70757b"/>',
                    f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" stroke="#70757b"/>'])
        for tick in (0.0, 0.05, 0.10, 0.15):
            y = plot_bottom - (tick / max_value) * (plot_bottom - plot_top)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width}" y2="{y:.1f}" stroke="#d9dadd"/>')
            svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#50565c">{tick:.2f}</text>')
        bar_width = 54
        for index, model in enumerate(models):
            value = lookup[(model, metric)]
            x = left + 48 + index * 88
            bar_height = (value / max_value) * (plot_bottom - plot_top)
            y = plot_bottom - bar_height
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[model]}"/>')
            svg.append(f'<text x="{x + bar_width / 2}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#20252b">{value:.3f}</text>')
            svg.append(f'<text x="{x + bar_width / 2}" y="{plot_bottom + 19}" text-anchor="middle" font-family="Arial" font-size="12" fill="#20252b">{model.split("_")[0]}</text>')
    svg.append('</svg>')
    output = output.with_suffix(".svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return True


def build(registry: Path, output_dir: Path) -> None:
    rows = load_rows(registry)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_rows = [row for row in rows if row["split"] == "test" and row["status"] == "primary"
                 and row["task"] in {"locked_test_horizon_risk", "locked_test_calibration"}
                 and row["metric"] in {"auc", "auprc", "brier", "ece"}]
    test_fields = ["model", "metric", "value", "dataset_profile", "horizon_years", "landmark_protocol", "censoring"]
    write_csv(output_dir / "locked_test_horizon_summary.csv",
              [{field: row.get(field, "") for field in test_fields} for row in test_rows], test_fields)

    validation = [row for row in rows if row["split"] == "val" and row["aggregation"] == "mean_across_seeds"
                  and row["status"] == "primary"]
    validation_fields = ["model", "result_class", "task", "metric", "value", "lower_ci", "upper_ci"]
    write_csv(output_dir / "validation_primary_summary.csv",
              [{field: row.get(field, "") for field in validation_fields} for row in validation], validation_fields)

    category_counts = []
    for result_class in sorted({row["result_class"] for row in rows}):
        class_rows = [row for row in rows if row["result_class"] == result_class]
        category_counts.append({"result_class": result_class, "records": str(len(class_rows)),
                                "primary": str(sum(row["status"] == "primary" for row in class_rows)),
                                "diagnostic_or_exploratory": str(sum(row["status"] in {"diagnostic", "exploratory"} for row in class_rows))})
    write_csv(output_dir / "registry_category_counts.csv", category_counts,
              ["result_class", "records", "primary", "diagnostic_or_exploratory"])

    md = [
        "# Compact Result Registry Views", "",
        "The JSONL/CSV registry is the audit ledger. This file is the human-facing summary.", "",
        "## Registry inventory", "",
    ]
    md.extend(markdown_table(category_counts, ["result_class", "records", "primary", "diagnostic_or_exploratory"]))
    md.extend(["", "## Locked test horizon head", "",
               "Only the frozen exact-target test rows are shown here; paired deltas remain in the ledger.", ""])
    md.extend(markdown_table(test_rows, test_fields))
    md.extend(["", "## Validation primary aggregates", ""])
    for result_class in ("lm_logits", "horizon_head", "other_medical"):
        class_rows = [row for row in validation if row["result_class"] == result_class]
        if not class_rows:
            continue
        md.append(f"### {result_class}")
        md.append("")
        md.extend(markdown_table(class_rows, validation_fields))
        md.append("")
    md.extend(["## Excluded from default ranking", "",
               "Legacy/non-censoring diagnostic rows, exploratory Top-K rows, and literature references remain in the ledger but are not mixed into the primary tables.", ""])
    (output_dir / "registry_compact_summary.md").write_text("\n".join(md), encoding="utf-8")

    figure_ok = make_figure(test_rows, output_dir / "locked_test_horizon_auc_auprc.png")
    (output_dir / "summary_manifest.json").write_text(json.dumps({
        "registry": str(registry), "record_count": len(rows), "test_primary_rows": len(test_rows),
        "validation_primary_rows": len(validation), "figure_generated": figure_ok,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote compact summaries to {output_dir}; figure_generated={figure_ok}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "results/paper_protocol_v1/result_registry/result_registry.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_protocol_v1/result_registry/compact")
    args = parser.parse_args()
    build(args.registry, args.output_dir)


if __name__ == "__main__":
    main()
