from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

from semantic_delphi_ukb.selected_disease_modern_eval import (
    read_csv_rows,
    write_comparison_md,
    write_modern_bar_svg,
)
from semantic_delphi_ukb.selected_disease_demo import evaluate_loaded_model, parse_selected_diseases, parse_topk, write_csv
from semantic_delphi_ukb.tte_selected_utils import load_tte_model, tte_checkpoint_meta


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate selected disease groups for the TTE multitask checkpoint.")
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "tests" / "output" / "selected_disease_tte_multitask")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument(
        "--baseline-metrics-csv",
        type=Path,
        default=REPO_DIR / "tests" / "output" / "selected_disease_modern_baseline" / "selected_disease_metrics.csv",
    )
    return parser


def write_tte_comparison_md(path: Path, modern_rows: Sequence[dict], tte_rows: Sequence[dict]) -> None:
    modern = {row["disease_id"]: row for row in modern_rows}
    lines = [
        "# TTE Multitask Selected-Disease Evaluation",
        "",
        "| Disease | Positives | Modern AUC | TTE AUC | Delta AUC | Modern Top10 | TTE Top10 | Delta Top10 | Modern Top-decile | TTE Top-decile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tte_rows:
        old = modern.get(row["disease_id"], {})
        lines.append(
            "| {name} | {pos} | {old_auc:.4f} | {new_auc:.4f} | {auc_delta:+.4f} | "
            "{old_top10:.4f} | {new_top10:.4f} | {top10_delta:+.4f} | {old_cap:.4f} | {new_cap:.4f} |".format(
                name=row.get("name", row["disease_id"]),
                pos=row.get("positives", ""),
                old_auc=float(old.get("auc", "nan")),
                new_auc=float(row.get("auc", "nan")),
                auc_delta=float(row.get("auc", "nan")) - float(old.get("auc", "nan")),
                old_top10=float(old.get("group_top10", "nan")),
                new_top10=float(row.get("group_top10", "nan")),
                top10_delta=float(row.get("group_top10", "nan")) - float(old.get("group_top10", "nan")),
                old_cap=float(old.get("top_decile_capture", "nan")),
                new_cap=float(row.get("top_decile_capture", "nan")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diseases = parse_selected_diseases(args.diseases_yaml)
    topk_values = parse_topk(args.topk)
    loaded = load_tte_model(split=args.split, device=args.device)
    metrics, bands = evaluate_loaded_model(
        loaded=loaded,
        diseases=diseases,
        batch_size=args.batch_size,
        max_patients=args.max_patients,
        topk_values=topk_values,
        device=args.device,
    )
    write_csv(args.output_dir / "selected_disease_metrics.csv", metrics)
    write_csv(args.output_dir / "selected_disease_risk_bands.csv", bands)
    (args.output_dir / "selected_disease_details.json").write_text(
        json.dumps(
            {
                "checkpoint": tte_checkpoint_meta(),
                "diseases": [d.__dict__ for d in diseases],
                "metrics": metrics,
                "risk_bands": bands,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_modern_bar_svg(args.output_dir / "selected_disease_auc.svg", metrics, "auc", "TTE selected-disease AUC")
    write_modern_bar_svg(
        args.output_dir / "selected_disease_top_decile_capture.svg",
        metrics,
        "top_decile_capture",
        "TTE top-decile capture",
    )
    baseline_rows = read_csv_rows(args.baseline_metrics_csv)
    write_tte_comparison_md(args.output_dir / "tte_vs_modern_selected_disease.md", baseline_rows, metrics)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(metrics), "checkpoint": tte_checkpoint_meta()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
