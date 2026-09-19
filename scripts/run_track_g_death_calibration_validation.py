"""Fit and evaluate validation-only temperature scaling for rollout death probabilities."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODELS = ("A0", "A2", "ETHOS-Matched", "Foresight-Matched")
SEEDS = (42, 43, 44)
CALIBRATION_VERSION = "track_g_death_calibration_v1"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "results/track_g_v1/val")
    parser.add_argument("--out-dir", type=Path, default=root / "results/track_g_v1/val_death_calibration")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def split_role(patient_index: int) -> str:
    digest = hashlib.sha256(f"{CALIBRATION_VERSION}:{patient_index}".encode()).digest()
    return "fit" if int.from_bytes(digest[:8], "big") % 2 == 0 else "eval"


def clipped_logits(probabilities: np.ndarray) -> np.ndarray:
    epsilon = 1.0e-4
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def scaled_probability(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = clipped_logits(probabilities) / float(temperature)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1.0e-7, 1.0 - 1.0e-7)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.exp(np.linspace(-4.0, 4.0, 1601))
    losses = np.asarray([binary_log_loss(scaled_probability(probabilities, t), labels) for t in candidates])
    best = np.flatnonzero(losses == losses.min())
    return float(candidates[int(best[0])])


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    rows = []
    total = len(labels)
    score = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (probabilities >= low) & (probabilities < high if index < bins - 1 else probabilities <= high)
        count = int(mask.sum())
        if not count:
            rows.append({"bin": index, "lower": low, "upper": high, "count": 0, "mean_probability": None, "observed_rate": None})
            continue
        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(labels[mask].mean())
        score += count / total * abs(mean_probability - observed_rate)
        rows.append({"bin": index, "lower": low, "upper": high, "count": count, "mean_probability": mean_probability, "observed_rate": observed_rate})
    return float(score), rows


def metric_row(model: str, seed: int, split: str, variant: str, probabilities: np.ndarray, labels: np.ndarray, temperature: float | None) -> tuple[dict, list[dict]]:
    ece_value, bins = ece(probabilities, labels)
    return ({
        "model": model,
        "seed": seed,
        "split": split,
        "variant": variant,
        "temperature": temperature,
        "patient_count": int(len(labels)),
        "observed_death_rate": float(labels.mean()),
        "mean_probability": float(probabilities.mean()),
        "brier": brier(probabilities, labels),
        "log_loss": binary_log_loss(probabilities, labels),
        "ece": ece_value,
    }, bins)


def bootstrap_delta(raw: np.ndarray, calibrated: np.ndarray, labels: np.ndarray, replicates: int, seed: int) -> dict:
    observed = float(np.mean((calibrated - labels) ** 2) - np.mean((raw - labels) ** 2))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(labels), size=(replicates, len(labels)))
    values = np.mean((calibrated[draws] - labels[draws]) ** 2, axis=1) - np.mean((raw[draws] - labels[draws]) ** 2, axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"delta_calibrated_minus_raw_brier": observed, "ci95": [float(low), float(high)]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    calibration_bins = []
    aggregate: dict[str, dict] = {model: {"seed_results": []} for model in MODELS}
    split_hash = hashlib.sha256("\n".join(f"{index}:{split_role(index)}" for index in range(10000)).encode()).hexdigest()

    for seed in SEEDS:
        for model in MODELS:
            path = args.input_root / f"seed{seed}" / model / "patient_rows.json.gz"
            rows = load_rows(path)
            rows.sort(key=lambda row: int(row["patient_index"]))
            probabilities = np.asarray([float(row["death_probability"]) for row in rows], dtype=float)
            labels = np.asarray([int(row["actual_death"]) for row in rows], dtype=float)
            roles = np.asarray([split_role(int(row["patient_index"])) for row in rows])
            fit_mask = roles == "fit"
            eval_mask = roles == "eval"
            temperature = fit_temperature(probabilities[fit_mask], labels[fit_mask])
            calibrated = scaled_probability(probabilities, temperature)
            raw_row, raw_bins = metric_row(model, seed, "eval", "raw", probabilities[eval_mask], labels[eval_mask], None)
            calibrated_row, calibrated_bins = metric_row(model, seed, "eval", "temperature_scaled", calibrated[eval_mask], labels[eval_mask], temperature)
            fit_row, _ = metric_row(model, seed, "fit", "raw", probabilities[fit_mask], labels[fit_mask], None)
            summary_rows.extend((fit_row, raw_row, calibrated_row))
            for bin_row in raw_bins:
                calibration_bins.append({**bin_row, "model": model, "seed": seed, "split": "eval", "variant": "raw"})
            for bin_row in calibrated_bins:
                calibration_bins.append({**bin_row, "model": model, "seed": seed, "split": "eval", "variant": "temperature_scaled"})
            aggregate[model]["seed_results"].append({
                "seed": seed,
                "temperature": temperature,
                "fit_patient_count": int(fit_mask.sum()),
                "eval_patient_count": int(eval_mask.sum()),
                "raw": raw_row,
                "temperature_scaled": calibrated_row,
                "brier_delta_bootstrap": bootstrap_delta(probabilities[eval_mask], calibrated[eval_mask], labels[eval_mask], args.bootstrap_replicates, args.bootstrap_seed + seed),
            })

    for model, payload in aggregate.items():
        for variant in ("raw", "temperature_scaled"):
            values = [item[variant] for item in payload["seed_results"]]
            payload[variant] = {
                metric: float(np.mean([row[metric] for row in values]))
                for metric in ("brier", "log_loss", "ece", "mean_probability", "observed_death_rate")
            }
        payload["temperature_mean"] = float(np.mean([item["temperature"] for item in payload["seed_results"]]))

    write_csv(args.out_dir / "metrics.csv", summary_rows)
    write_csv(args.out_dir / "reliability_bins.csv", calibration_bins)
    (args.out_dir / "split_manifest.json").write_text(json.dumps({
        "calibration_version": CALIBRATION_VERSION,
        "split_rule": "sha256(f'{version}:{patient_index}') first 8 bytes parity",
        "fit_role": "even parity",
        "eval_role": "odd parity",
        "split_hash_first_10000": split_hash,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "method": "temperature_scaling_fit_on_validation_fit_split_evaluated_on_validation_eval_split",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps({
        "calibration_version": CALIBRATION_VERSION,
        "input_root": str(args.input_root),
        "output_root": str(args.out_dir),
        "aggregate": aggregate,
        "rows": summary_rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
