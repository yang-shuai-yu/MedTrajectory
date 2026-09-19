from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_DIR / "results" / "locked_test_horizon_risk"


DEFAULT_RUNS = {
    "Monotonic Gated RoPE": "monotonic_gated_rope_gate100",
    "Gated RoPE main": "gated_rope_gate100",
    "No-RoPE previous best": "high_priority_no_rope_trunk_full3000",
    "MedTrajectory horizon risk": "medtrajectory_test",
    "BERT risk head": "bert_test",
    "Mamba risk head": "mamba_test",
    "Survival-horizon": "survival_test",
}


def read_raw_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def summarize_model(model_name: str, raw_path: Path, ks: tuple[int, ...]) -> list[dict]:
    rows = read_raw_predictions(raw_path)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["patient_index"], row["prediction_age_years"], row["horizon_years"])].append(row)

    buckets: dict[float, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (_, _, horizon), items in grouped.items():
        items = sorted(items, key=lambda row: safe_float(row["score"]), reverse=True)
        positives = [row for row in items if int(float(row["label"])) == 1]
        if not positives:
            continue
        horizon_value = float(horizon)
        buckets[horizon_value]["patient_horizon_with_positive"] += 1
        buckets[horizon_value]["positive_labels"] += len(positives)
        rank_by_disease = {row["disease_id"]: idx + 1 for idx, row in enumerate(items)}
        positive_ranks = [rank_by_disease[row["disease_id"]] for row in positives]
        best_rank = min(positive_ranks)
        buckets[horizon_value]["mrr_sum"] += 1.0 / best_rank
        buckets[horizon_value]["mean_best_rank_sum"] += best_rank
        for k in ks:
            hit = float(any(rank <= k for rank in positive_ranks))
            recall = sum(1 for rank in positive_ranks if rank <= k) / len(positive_ranks)
            buckets[horizon_value][f"top{k}_any_hit_sum"] += hit
            buckets[horizon_value][f"top{k}_label_recall_sum"] += recall

    out = []
    for horizon, stats in sorted(buckets.items()):
        denom = max(1.0, stats["patient_horizon_with_positive"])
        row = {
            "model": model_name,
            "horizon_years": horizon,
            "patient_horizon_with_positive": int(stats["patient_horizon_with_positive"]),
            "positive_labels": int(stats["positive_labels"]),
            "mrr_any_positive": stats["mrr_sum"] / denom,
            "mean_best_rank": stats["mean_best_rank_sum"] / denom,
        }
        for k in ks:
            row[f"top{k}_any_positive_hit"] = stats[f"top{k}_any_hit_sum"] / denom
            row[f"top{k}_positive_label_recall"] = stats[f"top{k}_label_recall_sum"] / denom
        out.append(row)

    if out:
        overall = {"model": model_name, "horizon_years": "overall"}
        total_den = sum(r["patient_horizon_with_positive"] for r in out)
        total_pos = sum(r["positive_labels"] for r in out)
        overall["patient_horizon_with_positive"] = total_den
        overall["positive_labels"] = total_pos
        for metric in ["mrr_any_positive", "mean_best_rank"]:
            overall[metric] = sum(r[metric] * r["patient_horizon_with_positive"] for r in out) / max(1, total_den)
        for k in ks:
            for suffix in ["any_positive_hit", "positive_label_recall"]:
                key = f"top{k}_{suffix}"
                overall[key] = sum(r[key] * r["patient_horizon_with_positive"] for r in out) / max(1, total_den)
        out.append(overall)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize selected-disease Top-K hit metrics from horizon risk raw predictions.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR / "topk_disease_hit_comparison")
    parser.add_argument("--ks", type=str, default="1,3,5,10")
    args = parser.parse_args()

    ks = tuple(sorted({int(item.strip()) for item in args.ks.split(",") if item.strip()}))
    all_rows = []
    missing = {}
    for model_name, run_dir in DEFAULT_RUNS.items():
        raw_path = args.results_dir / run_dir / "raw_predictions.csv"
        if not raw_path.exists():
            missing[model_name] = str(raw_path)
            continue
        all_rows.extend(summarize_model(model_name, raw_path, ks))

    write_csv(args.out_dir / "horizon_selected_disease_topk_comparison.csv", all_rows)
    payload = {"ks": ks, "rows": all_rows, "missing": missing}
    (args.out_dir / "horizon_selected_disease_topk_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.out_dir / "horizon_selected_disease_topk_comparison.csv")
    if missing:
        print(json.dumps({"missing": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
