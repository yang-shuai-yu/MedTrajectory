from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_DIR / "results" / "expanded_horizon_risk"


DEFAULT_MODELS = {
    "MedTrajectory monotonic 38d full3000": DEFAULT_ROOT / "medtrajectory_monotonic_38d_full3000_test" / "risk_metrics_rows.csv",
    "MedTrajectory monotonic 38d smoke1000": DEFAULT_ROOT / "medtrajectory_monotonic_38d_smoke1000_test" / "risk_metrics_rows.csv",
    "BERT 38d full3000": DEFAULT_ROOT / "bert_38d_full3000_test" / "risk_metrics_rows.csv",
    "BERT 38d smoke1000": DEFAULT_ROOT / "bert_38d_smoke1000_test" / "risk_metrics_rows.csv",
    "Mamba 38d full3000": DEFAULT_ROOT / "mamba_38d_full3000_test" / "risk_metrics_rows.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize expanded horizon-risk metrics by reporting tier.")
    parser.add_argument("--counts-wide", type=Path, default=REPO_DIR / "results" / "expanded_disease_panel" / "expanded_disease_panel_counts_wide.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "comparison")
    return parser


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean(values: list[float]) -> float:
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    tiers = {row["disease_id"]: row for row in read_csv(args.counts_wide)}
    detailed_rows = []
    for model_name, path in DEFAULT_MODELS.items():
        if not path.exists():
            continue
        for row in read_csv(path):
            tier = tiers.get(row["disease_id"], {})
            merged = {
                "model": model_name,
                **row,
                "reporting_tier": tier.get("reporting_tier", ""),
                "test_5y_final_context_positives": tier.get("test_5y_final_context_positives", ""),
                "test_10y_final_context_positives": tier.get("test_10y_final_context_positives", ""),
                "token_count": tier.get("token_count", ""),
            }
            detailed_rows.append(merged)

    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in detailed_rows:
        if np.isnan(safe_float(row.get("auc"))):
            continue
        horizon = str(int(float(row["horizon_years"])))
        buckets[(row["model"], row["reporting_tier"], horizon)].append(row)
        buckets[(row["model"], row["reporting_tier"], "overall")].append(row)
        if row["reporting_tier"] in {"headline", "exploratory"}:
            buckets[(row["model"], "headline_plus_exploratory", horizon)].append(row)
            buckets[(row["model"], "headline_plus_exploratory", "overall")].append(row)

    summary_rows = []
    for (model, tier, horizon), rows in sorted(buckets.items()):
        summary_rows.append(
            {
                "model": model,
                "reporting_tier": tier,
                "horizon_years": horizon,
                "diseases_with_auc": len(rows),
                "auc_mean": mean([safe_float(r["auc"]) for r in rows]),
                "average_precision_mean": mean([safe_float(r["average_precision"]) for r in rows]),
                "brier_mean": mean([safe_float(r["brier"]) for r in rows]),
                "ece_mean": mean([safe_float(r["ece"]) for r in rows]),
                "top_decile_capture_mean": mean([safe_float(r["top_decile_capture"]) for r in rows]),
                "event_rate_mean": mean([safe_float(r["event_rate"]) for r in rows]),
                "positives_sum": int(sum(safe_float(r["positives"]) for r in rows)),
            }
        )
    write_csv(args.out_dir / "expanded_horizon_risk_detailed_with_tiers.csv", detailed_rows)
    write_csv(args.out_dir / "expanded_horizon_risk_tier_summary.csv", summary_rows)
    print(args.out_dir / "expanded_horizon_risk_tier_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
