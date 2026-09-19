from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_DIR / "results" / "locked_test_horizon_risk"


DEEP_MODEL_SPECS = [
    ("Monotonic Gated RoPE", LOCKED_DIR / "monotonic_gated_rope_gate100"),
    ("Gated RoPE main", LOCKED_DIR / "gated_rope_gate100"),
    ("No-RoPE previous best", LOCKED_DIR / "high_priority_no_rope_trunk_full3000"),
    ("Old TTE+RoPE main", LOCKED_DIR / "high_priority_tte_trunk_aux02"),
    ("BERT risk head", LOCKED_DIR / "bert_test"),
    ("Mamba risk head", LOCKED_DIR / "mamba_test"),
]


ML_RAW = LOCKED_DIR / "frozen_ml_baselines" / "frozen_ml_raw_predictions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare horizon-risk metrics that reflect age-varying biomedical state.")
    parser.add_argument("--locked-dir", type=Path, default=LOCKED_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "age_state_evaluation")
    parser.add_argument("--include-ml", action="store_true", help="Include Static/Dynamic ML raw predictions if present.")
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    positives = int(labels_bool.sum())
    negatives = int(labels_bool.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[labels_bool].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    hit_ranks = np.flatnonzero(ranked) + 1
    return float((np.arange(1, positives + 1, dtype=np.float64) / hit_ranks).mean())


def brier(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2)) if labels.size else float("nan")


def ece(scores: np.ndarray, labels: np.ndarray, bins: int) -> float:
    if labels.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (scores >= edges[idx]) & (scores <= edges[idx + 1])
        else:
            mask = (scores >= edges[idx]) & (scores < edges[idx + 1])
        if bool(mask.any()):
            value += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(value)


def top_decile_stats(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    if labels.size == 0 or labels.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(math.ceil(labels.size * 0.10)))
    top_labels = labels[order[:top_n]]
    top_rate = float(top_labels.mean())
    baseline = float(labels.mean())
    return float(top_labels.sum() / labels.sum()), top_rate, top_rate / baseline if baseline else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def age_bin(age_years: float) -> str:
    if age_years < 50.0:
        return "<50"
    if age_years < 60.0:
        return "50-59"
    if age_years < 70.0:
        return "60-69"
    return "70+"


def load_deep_predictions(locked_dir: Path) -> list[dict]:
    rows = []
    specs = [(name, locked_dir / rel.name) for name, rel in DEEP_MODEL_SPECS]
    for model_name, directory in specs:
        for row in read_csv(directory / "raw_predictions.csv"):
            item = dict(row)
            item["model"] = model_name
            item["source_dir"] = str(directory.relative_to(REPO_DIR))
            rows.append(item)
    return rows


def load_ml_predictions(path: Path) -> list[dict]:
    rows = []
    for row in read_csv(path):
        item = dict(row)
        item["source_dir"] = str(path.parent.relative_to(REPO_DIR))
        rows.append(item)
    return rows


def metric_row(model: str, horizon: str, rows: list[dict], bins: int) -> dict:
    scores = np.asarray([as_float(row["score"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(float(row["label"])) for row in rows], dtype=np.int8)
    ages = np.asarray([as_float(row["prediction_age_years"]) for row in rows], dtype=np.float64)
    capture, top_rate, lift = top_decile_stats(scores, labels)
    return {
        "model": model,
        "horizon_years": horizon,
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "event_rate": float(labels.mean()) if labels.size else float("nan"),
        "mean_score": float(scores.mean()) if scores.size else float("nan"),
        "auc_micro": binary_auc(scores, labels),
        "average_precision_micro": average_precision(scores, labels),
        "brier_micro": brier(scores, labels),
        "ece_micro": ece(scores, labels, bins),
        "top_decile_capture_micro": capture,
        "top_decile_event_rate_micro": top_rate,
        "top_decile_lift_micro": lift,
        "score_age_corr": pearson(ages, scores),
        "label_age_corr": pearson(ages, labels.astype(np.float64)),
    }


def build_overall_rows(rows: list[dict], bins: int) -> list[dict]:
    out = []
    by_model_horizon: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_model_horizon[(row["model"], f"{as_float(row['horizon_years']):g}")].append(row)
    for (model, horizon), group in sorted(by_model_horizon.items()):
        out.append(metric_row(model, horizon, group, bins))
    return out


def build_age_bin_rows(rows: list[dict], bins: int) -> list[dict]:
    out = []
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["model"], f"{as_float(row['horizon_years']):g}", age_bin(as_float(row["prediction_age_years"])))].append(row)
    for (model, horizon, bin_name), group in sorted(by_key.items()):
        result = metric_row(model, horizon, group, bins)
        result["age_bin"] = bin_name
        out.append(result)
    return out


def build_horizon_consistency_rows(rows: list[dict]) -> list[dict]:
    by_model: dict[str, dict[tuple[str, str], dict[str, tuple[float, int]]]] = defaultdict(dict)
    for row in rows:
        key = (row["patient_index"], row["disease_id"])
        horizon = f"{as_float(row['horizon_years']):g}"
        by_model[row["model"]].setdefault(key, {})[horizon] = (as_float(row["score"]), int(float(row["label"])))

    out = []
    for model, by_key in sorted(by_model.items()):
        score_deltas = []
        label_deltas = []
        violations = 0
        label_violations = 0
        for values in by_key.values():
            if "5" not in values or "10" not in values:
                continue
            score5, label5 = values["5"]
            score10, label10 = values["10"]
            delta = score10 - score5
            score_deltas.append(delta)
            label_deltas.append(label10 - label5)
            if delta < -1e-12:
                violations += 1
            if label10 < label5:
                label_violations += 1
        deltas = np.asarray(score_deltas, dtype=np.float64)
        out.append(
            {
                "model": model,
                "paired_patient_disease_rows": int(deltas.size),
                "horizon_monotonic_rate_score10_ge_score5": float(1.0 - violations / deltas.size) if deltas.size else float("nan"),
                "horizon_violation_rate": float(violations / deltas.size) if deltas.size else float("nan"),
                "mean_score_10y_minus_5y": float(deltas.mean()) if deltas.size else float("nan"),
                "median_score_10y_minus_5y": float(np.median(deltas)) if deltas.size else float("nan"),
                "label_monotonic_rate": float(1.0 - label_violations / deltas.size) if deltas.size else float("nan"),
                "mean_label_10y_minus_5y": float(np.mean(label_deltas)) if label_deltas else float("nan"),
            }
        )
    return out


def build_macro_rows(locked_dir: Path, include_ml: bool) -> list[dict]:
    rows = []
    for model_name, directory in [(name, locked_dir / rel.name) for name, rel in DEEP_MODEL_SPECS]:
        aggregate = read_csv(directory / "aggregate_metrics.csv")
        overall = next((row for row in aggregate if row.get("aggregate") == "overall"), None)
        h5 = next((row for row in aggregate if as_float(row.get("horizon_years")) == 5.0), None)
        h10 = next((row for row in aggregate if as_float(row.get("horizon_years")) == 10.0), None)
        if overall is None:
            continue
        rows.append(
            {
                "model": model_name,
                "auc_5y_macro": as_float((h5 or {}).get("auc_mean")),
                "auc_10y_macro": as_float((h10 or {}).get("auc_mean")),
                "auc_overall_macro": as_float(overall.get("auc_mean")),
                "average_precision_macro": as_float(overall.get("average_precision_mean")),
                "brier_macro": as_float(overall.get("brier_mean")),
                "ece_macro": as_float(overall.get("ece_mean")),
                "top_decile_capture_macro": as_float(overall.get("top_decile_capture_mean")),
                "source_dir": str(directory.relative_to(REPO_DIR)),
            }
        )
    if include_ml:
        aggregate = read_csv(locked_dir / "frozen_ml_baselines" / "frozen_ml_aggregate_metrics.csv")
        for model_name in sorted({row["model"] for row in aggregate if row.get("model")}):
            model_rows = [row for row in aggregate if row.get("model") == model_name]
            overall = next((row for row in model_rows if row.get("aggregate") == "overall"), None)
            h5 = next((row for row in model_rows if as_float(row.get("horizon_years")) == 5.0), None)
            h10 = next((row for row in model_rows if as_float(row.get("horizon_years")) == 10.0), None)
            if overall is None:
                continue
            rows.append(
                {
                    "model": model_name,
                    "auc_5y_macro": as_float((h5 or {}).get("auc_mean")),
                    "auc_10y_macro": as_float((h10 or {}).get("auc_mean")),
                    "auc_overall_macro": as_float(overall.get("auc_mean")),
                    "average_precision_macro": as_float(overall.get("average_precision_mean")),
                    "brier_macro": as_float(overall.get("brier_mean")),
                    "ece_macro": as_float(overall.get("ece_mean")),
                    "top_decile_capture_macro": as_float(overall.get("top_decile_capture_mean")),
                    "source_dir": str((locked_dir / "frozen_ml_baselines").relative_to(REPO_DIR)),
                }
            )
    return sorted(rows, key=lambda row: as_float(row["auc_overall_macro"]), reverse=True)


def write_status(path: Path, macro_rows: list[dict], consistency_rows: list[dict]) -> None:
    lines = [
        "# Age-State Horizon Risk Evaluation",
        "",
        "This post-hoc table compares not only AUC, but also calibration, risk concentration, age association, and 5y/10y horizon consistency. It is intended to support the claim that the model can represent an age-varying biomedical state rather than only classify a fixed endpoint.",
        "",
        "## Macro Locked-Test Metrics",
        "",
        "| Model | 5y AUC | 10y AUC | Overall AUC | AP | Brier | ECE | Top-decile capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macro_rows:
        lines.append(
            "| {model} | {auc5:.4f} | {auc10:.4f} | {auc:.4f} | {ap:.4f} | {brier:.4f} | {ece:.4f} | {capture:.4f} |".format(
                model=row["model"],
                auc5=as_float(row["auc_5y_macro"]),
                auc10=as_float(row["auc_10y_macro"]),
                auc=as_float(row["auc_overall_macro"]),
                ap=as_float(row["average_precision_macro"]),
                brier=as_float(row["brier_macro"]),
                ece=as_float(row["ece_macro"]),
                capture=as_float(row["top_decile_capture_macro"]),
            )
        )
    lines.extend(
        [
            "",
            "## Horizon Consistency",
            "",
            "| Model | 10y>=5y score rate | Mean 10y-5y score | Violation rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in consistency_rows:
        lines.append(
            "| {model} | {rate:.4f} | {delta:.6f} | {viol:.4f} |".format(
                model=row["model"],
                rate=as_float(row["horizon_monotonic_rate_score10_ge_score5"]),
                delta=as_float(row["mean_score_10y_minus_5y"]),
                viol=as_float(row["horizon_violation_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_deep_predictions(args.locked_dir)
    if args.include_ml:
        rows.extend(load_ml_predictions(args.locked_dir / "frozen_ml_baselines" / "frozen_ml_raw_predictions.csv"))
    macro_rows = build_macro_rows(args.locked_dir, args.include_ml)
    overall_rows = build_overall_rows(rows, args.calibration_bins)
    age_rows = build_age_bin_rows(rows, args.calibration_bins)
    consistency_rows = build_horizon_consistency_rows(rows)
    write_csv(args.out_dir / "risk_macro_comparison.csv", macro_rows)
    write_csv(args.out_dir / "risk_micro_comparison.csv", overall_rows)
    write_csv(args.out_dir / "risk_age_bin_comparison.csv", age_rows)
    write_csv(args.out_dir / "risk_horizon_consistency.csv", consistency_rows)
    write_status(args.out_dir / "STATUS.md", macro_rows, consistency_rows)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
