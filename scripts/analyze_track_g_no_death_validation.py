"""Paired validation analysis of raw versus no-death-token trajectories."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


MODELS = ("A0", "A2")
SEEDS = (42, 43, 44)
METRICS = {
    "diagnosis_jaccard": {"subset": "all", "direction": "higher"},
    "hit_at_10": {"subset": "nondeath_first_target", "direction": "higher"},
    "first_event_time_mae_days": {"subset": "no_observed_death", "direction": "lower"},
    "event_count_mae": {"subset": "no_observed_death", "direction": "lower"},
    "waiting_time_nll": {"subset": "nondeath_first_target", "direction": "lower"},
}


def load_rows(path: Path) -> dict[int, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    result = {int(row["patient_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate patient rows in {path}")
    return result


def eligible(row: dict, subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "nondeath_first_target":
        return row["target_event_type"] != "death"
    if subset == "no_observed_death":
        return int(row["actual_death"]) == 0
    raise ValueError(subset)


def aggregate_pair(raw_root: Path, ablation_root: Path, model: str, metric: str, subset: str):
    by_patient: dict[int, list[tuple[float, float]]] = {}
    seed_counts = {}
    for seed in SEEDS:
        raw = load_rows(raw_root / f"seed{seed}" / model / "patient_rows.json.gz")
        ablation = load_rows(ablation_root / f"seed{seed}" / model / "patient_rows.json.gz")
        if raw.keys() != ablation.keys():
            raise ValueError(f"patient set mismatch for {model} seed {seed}")
        seed_counts[str(seed)] = len(raw)
        for patient_index, raw_row in raw.items():
            if not eligible(raw_row, subset):
                continue
            raw_value = raw_row.get(metric)
            ablation_value = ablation[patient_index].get(metric)
            if raw_value is None or ablation_value is None:
                continue
            if not math.isfinite(float(raw_value)) or not math.isfinite(float(ablation_value)):
                continue
            by_patient.setdefault(patient_index, []).append((float(raw_value), float(ablation_value)))
    patients = np.asarray(sorted(by_patient), dtype=np.int64)
    raw = np.asarray([np.mean([pair[0] for pair in by_patient[int(pid)]]) for pid in patients])
    ablation = np.asarray([np.mean([pair[1] for pair in by_patient[int(pid)]]) for pid in patients])
    return patients, raw, ablation, seed_counts


def bootstrap_delta(raw: np.ndarray, ablation: np.ndarray, replicates: int, seed: int) -> dict:
    delta = ablation - raw
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(delta), size=(replicates, len(delta)))
    values = delta[sampled].mean(axis=1)
    return {
        "raw_mean": float(raw.mean()),
        "no_death_mean": float(ablation.mean()),
        "delta_no_death_minus_raw": float(delta.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "patient_count": int(len(delta)),
        "bootstrap_replicates": int(replicates),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=root / "results/track_g_v1/val")
    parser.add_argument("--ablation-root", type=Path, default=root / "results/track_g_v1/val_no_death_token")
    parser.add_argument("--output", type=Path, default=root / "results/track_g_v1/val_no_death_token/paired_assessment.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    args = parser.parse_args()

    output = {
        "analysis": "validation_only_raw_vs_no_death_token",
        "models": {},
        "metric_subset_contract": METRICS,
    }
    ablation_values: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {model: {} for model in MODELS}
    for model_index, model in enumerate(MODELS):
        output["models"][model] = {}
        for metric_index, (metric, contract) in enumerate(METRICS.items()):
            patients, raw, ablation, seed_counts = aggregate_pair(
                args.raw_root, args.ablation_root, model, metric, contract["subset"]
            )
            result = bootstrap_delta(
                raw,
                ablation,
                args.bootstrap_replicates,
                args.bootstrap_seed + model_index * 100 + metric_index,
            )
            result["direction"] = contract["direction"]
            result["seed_patient_counts"] = seed_counts
            output["models"][model][metric] = result
            ablation_values[model][metric] = (patients, ablation)

    output["no_death_a2_minus_a0"] = {}
    for metric_index, (metric, contract) in enumerate(METRICS.items()):
        a0_patients, a0 = ablation_values["A0"][metric]
        a2_patients, a2 = ablation_values["A2"][metric]
        common = np.intersect1d(a0_patients, a2_patients)
        a0_map = dict(zip(a0_patients.tolist(), a0.tolist()))
        a2_map = dict(zip(a2_patients.tolist(), a2.tolist()))
        left = np.asarray([a0_map[int(pid)] for pid in common])
        right = np.asarray([a2_map[int(pid)] for pid in common])
        result = bootstrap_delta(left, right, args.bootstrap_replicates, args.bootstrap_seed + 1000 + metric_index)
        result["delta_name"] = "A2_minus_A0"
        result["direction"] = contract["direction"]
        output["no_death_a2_minus_a0"][metric] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
