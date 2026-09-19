from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_horizon_risk_locked_test import (
    average_precision,
    brier_score,
    calibration_stats,
    top_decile_event_count,
)
from semantic_delphi_ukb.train_architecture_risk_heads import binary_auc, top_decile_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit val-split Platt calibration and apply it to horizon-risk predictions.")
    parser.add_argument("--val-raw", type=Path, required=True)
    parser.add_argument("--test-raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=200)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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


def fit_platt(scores: np.ndarray, labels: np.ndarray, max_iter: int) -> tuple[float, float]:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1.0 - 1e-6)
    x = torch.tensor(np.log(clipped / (1.0 - clipped)), dtype=torch.float64)
    y = torch.tensor(labels.astype(np.float64), dtype=torch.float64)
    scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = scale * x + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scale.detach().cpu()), float(bias.detach().cpu())


def apply_platt(score: float, scale: float, bias: float) -> float:
    score = min(max(float(score), 1e-6), 1.0 - 1e-6)
    logit = math.log(score / (1.0 - score))
    z = scale * logit + bias
    return float(1.0 / (1.0 + math.exp(-z)))


def summarize(rows: list[dict], score_key: str, bins: int) -> tuple[list[dict], list[dict]]:
    diseases = sorted({row["disease_id"] for row in rows})
    horizons = sorted({float(row["horizon_years"]) for row in rows})
    summary_rows = []
    calibration_rows = []
    for horizon in horizons:
        for disease_id in diseases:
            subset = [row for row in rows if float(row["horizon_years"]) == horizon and row["disease_id"] == disease_id]
            scores = np.asarray([float(row[score_key]) for row in subset], dtype=np.float64)
            labels = np.asarray([int(row["label"]) for row in subset], dtype=np.int8)
            positives = int(labels.sum())
            negatives = int(labels.size - positives)
            auc = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
            capture, top_rate, lift = top_decile_stats(scores, labels)
            ece, cal_rows = calibration_stats(scores, labels, bins)
            name = subset[0]["name"] if subset else disease_id
            for cal in cal_rows:
                cal.update({"horizon_years": horizon, "disease_id": disease_id, "name": name})
                calibration_rows.append(cal)
            summary_rows.append(
                {
                    "horizon_years": horizon,
                    "disease_id": disease_id,
                    "name": name,
                    "prediction_moments": int(labels.size),
                    "positives": positives,
                    "negatives": negatives,
                    "auc": auc,
                    "average_precision": average_precision(scores, labels),
                    "brier": brier_score(scores, labels),
                    "ece": ece,
                    "top_decile_capture": capture,
                    "top_decile_event_rate": top_rate,
                    "top_decile_event_count": top_decile_event_count(scores, labels),
                    "top_decile_lift": lift,
                    "mean_score": float(scores.mean()) if scores.size else float("nan"),
                    "event_rate": float(labels.mean()) if labels.size else float("nan"),
                }
            )
    return summary_rows, calibration_rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    horizons = sorted({float(row["horizon_years"]) for row in rows})
    for horizon in horizons:
        subset = [row for row in rows if float(row["horizon_years"]) == horizon and not math.isnan(float(row["auc"]))]
        out.append(aggregate_row(subset, horizon, "horizon"))
    out.append(aggregate_row([row for row in rows if not math.isnan(float(row["auc"]))], float("nan"), "overall"))
    return out


def aggregate_row(rows: list[dict], horizon: float, scope: str) -> dict:
    def avg(key: str) -> float:
        values = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "aggregate": scope,
        "horizon_years": horizon,
        "diseases_with_auc": len(rows),
        "auc_mean": avg("auc"),
        "average_precision_mean": avg("average_precision"),
        "brier_mean": avg("brier"),
        "ece_mean": avg("ece"),
        "top_decile_capture_mean": avg("top_decile_capture"),
        "event_rate_mean": avg("event_rate"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_rows = read_rows(args.val_raw)
    test_rows = read_rows(args.test_raw)
    val_scores = np.asarray([float(row["score"]) for row in val_rows], dtype=np.float64)
    val_labels = np.asarray([int(row["label"]) for row in val_rows], dtype=np.int8)
    scale, bias = fit_platt(val_scores, val_labels, args.max_iter)

    calibrated_rows = []
    for row in test_rows:
        copied = dict(row)
        copied["score_uncalibrated"] = copied["score"]
        copied["score"] = apply_platt(float(copied["score"]), scale, bias)
        calibrated_rows.append(copied)

    summary_rows, calibration_rows = summarize(calibrated_rows, "score", args.calibration_bins)
    aggregate_rows = aggregate(summary_rows)
    write_csv(args.out_dir / "raw_predictions_calibrated.csv", calibrated_rows)
    write_csv(args.out_dir / "risk_metrics_rows_calibrated.csv", summary_rows)
    write_csv(args.out_dir / "aggregate_metrics_calibrated.csv", aggregate_rows)
    write_csv(args.out_dir / "calibration_bins_calibrated.csv", calibration_rows)
    (args.out_dir / "calibration_params.json").write_text(
        json.dumps({"scale": scale, "bias": bias, "val_raw": str(args.val_raw), "test_raw": str(args.test_raw)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"scale": scale, "bias": bias, "aggregates": aggregate_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
