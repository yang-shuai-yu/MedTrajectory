"""Aggregate validation-only death-hazard heads and compare with rollout risk."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("A0", "A2", "ETHOS-Matched", "Foresight-Matched")
SEEDS = (42, 43, 44)
HORIZONS = ("1y", "5y", "10y")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    n_pos = int(positive.sum()); n_neg = int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    hit = labels[np.argsort(-scores, kind="mergesort")].astype(bool)
    ranks = np.flatnonzero(hit) + 1
    return float((np.arange(1, positives + 1) / ranks).mean())


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & (scores <= high if index == bins - 1 else scores < high)
        if mask.any():
            result += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return result


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "n": int(len(labels)),
        "cases": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "mean_probability": float(scores.mean()),
        "auroc": auc(scores, labels),
        "auprc": average_precision(scores, labels),
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece": ece(scores, labels),
    }


def load_hazard_rows(root: Path, model: str) -> dict[int, dict]:
    patients: dict[int, dict] = {}
    for seed in SEEDS:
        path = root / f"seed{seed}" / model / "eval_rows.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                patient = int(row["patient_index"])
                item = patients.setdefault(patient, {
                    "labels": row["labels"], "masks": row["masks"], "raw": [], "calibrated": []
                })
                if item["labels"] != row["labels"] or item["masks"] != row["masks"]:
                    raise ValueError(f"inconsistent hazard target for {model} patient {patient}")
                item["raw"].append(row["raw_probability"])
                item["calibrated"].append(row["calibrated_probability"])
    return patients


def load_rollout(root: Path, model: str) -> dict[int, float]:
    values = defaultdict(list)
    for seed in SEEDS:
        path = root / f"seed{seed}" / model / "patient_rows.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                values[int(row["patient_index"])].append(float(row["death_probability"]))
    return {patient: float(np.mean(probabilities)) for patient, probabilities in values.items()}


def horizon_arrays(rows: dict[int, dict], index: int, key: str):
    patient_ids = np.asarray(sorted(patient for patient, row in rows.items() if row["masks"][index]), dtype=np.int64)
    labels = np.asarray([rows[int(patient)]["labels"][index] for patient in patient_ids], dtype=np.int8)
    scores = np.asarray([np.mean(rows[int(patient)][key], axis=0)[index] for patient in patient_ids], dtype=float)
    return patient_ids, scores, labels


def bootstrap_contrast(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    full = {
        "delta_brier": float(np.mean((right - labels) ** 2) - np.mean((left - labels) ** 2)),
        "delta_auroc": float(auc(right, labels) - auc(left, labels)),
        "delta_auprc": float(average_precision(right, labels) - average_precision(left, labels)),
    }
    rng = np.random.default_rng(seed)
    values = {key: [] for key in full}
    for _ in range(replicates):
        index = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[index]
        if sampled_labels.min() == sampled_labels.max():
            continue
        values["delta_brier"].append(float(np.mean((right[index] - sampled_labels) ** 2) - np.mean((left[index] - sampled_labels) ** 2)))
        values["delta_auroc"].append(float(auc(right[index], sampled_labels) - auc(left[index], sampled_labels)))
        values["delta_auprc"].append(float(average_precision(right[index], sampled_labels) - average_precision(left[index], sampled_labels)))
    for key, samples in values.items():
        full[f"{key}_ci95"] = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
    full["patient_count"] = int(len(labels))
    full["bootstrap_replicates"] = int(replicates)
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "results/track_g_v1/val_death_hazard_v2_followup")
    parser.add_argument("--rollout-root", type=Path, default=ROOT / "results/track_g_v1/val")
    parser.add_argument("--output", type=Path, default=ROOT / "results/track_g_v1/val_death_hazard_v2_followup/assessment.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    args = parser.parse_args()
    result = {"analysis": "validation_only_independent_death_hazard", "models": {}, "rollout_10y_comparison": {}}
    model_arrays = {}
    for model_index, model in enumerate(MODELS):
        rows = load_hazard_rows(args.input_root, model)
        result["models"][model] = {"raw": {}, "calibrated": {}}
        model_arrays[model] = {}
        for horizon_index, horizon in enumerate(HORIZONS):
            patient_ids, raw, labels = horizon_arrays(rows, horizon_index, "raw")
            calibrated_ids, calibrated, calibrated_labels = horizon_arrays(rows, horizon_index, "calibrated")
            if not np.array_equal(patient_ids, calibrated_ids) or not np.array_equal(labels, calibrated_labels):
                raise ValueError("raw/calibrated patient mismatch")
            result["models"][model]["raw"][horizon] = metrics(raw, labels)
            result["models"][model]["calibrated"][horizon] = metrics(calibrated, labels)
            model_arrays[model][horizon] = (patient_ids, calibrated, labels)

        rollout = load_rollout(args.rollout_root, model)
        patient_ids, head_score, labels = model_arrays[model]["10y"]
        rollout_score = np.asarray([rollout[int(patient)] for patient in patient_ids], dtype=float)
        result["rollout_10y_comparison"][model] = {
            "rollout": metrics(rollout_score, labels),
            "independent_head_calibrated": metrics(head_score, labels),
            "head_minus_rollout": bootstrap_contrast(
                labels, rollout_score, head_score, args.bootstrap_replicates,
                args.bootstrap_seed + 500 + model_index,
            ),
        }

    result["calibrated_a2_minus_comparators"] = {}
    for comparator_index, comparator in enumerate(("A0", "ETHOS-Matched", "Foresight-Matched")):
        result["calibrated_a2_minus_comparators"][comparator] = {}
        for horizon_index, horizon in enumerate(HORIZONS):
            a2_ids, a2_score, a2_labels = model_arrays["A2"][horizon]
            other_ids, other_score, other_labels = model_arrays[comparator][horizon]
            common = np.intersect1d(a2_ids, other_ids)
            a2_map = {int(pid): (float(score), int(label)) for pid, score, label in zip(a2_ids, a2_score, a2_labels)}
            other_map = {int(pid): (float(score), int(label)) for pid, score, label in zip(other_ids, other_score, other_labels)}
            labels = np.asarray([a2_map[int(pid)][1] for pid in common], dtype=np.int8)
            if any(other_map[int(pid)][1] != label for pid, label in zip(common, labels)):
                raise ValueError("model label mismatch")
            other = np.asarray([other_map[int(pid)][0] for pid in common])
            a2 = np.asarray([a2_map[int(pid)][0] for pid in common])
            result["calibrated_a2_minus_comparators"][comparator][horizon] = bootstrap_contrast(
                labels, other, a2, args.bootstrap_replicates,
                args.bootstrap_seed + comparator_index * 100 + horizon_index,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
