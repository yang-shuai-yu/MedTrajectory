"""Aggregate paired Track G patient rows across the three registered training seeds."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.track_g_contract import (  # noqa: E402
    assert_split_allowed,
    assert_training_allowed,
    load_track_g_protocol,
)
HEADLINE_METRICS = (
    "hit_at_10",
    "waiting_time_nll",
    "diagnosis_jaccard",
    "first_event_time_mae_days",
    "event_count_mae",
    "death_brier",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=REPO_DIR / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument("--split", choices=("val", "test"), default="val")
    value.add_argument("--input-root", type=Path, default=None)
    value.add_argument("--out-dir", type=Path, default=None)
    return value


def read_rows(path: Path) -> dict[int, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    output = {int(row["patient_index"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate patient_index in {path}")
    return output


def finite(value) -> bool:
    return value is not None and math.isfinite(value)


def bootstrap_delta(left, right, patient_ids, metric, replicates, seed):
    rng = np.random.default_rng(seed)
    training_seeds = sorted(left)
    metric_patient_ids = [
        int(pid)
        for pid in patient_ids
        if all(
            finite(left[training_seed][int(pid)][metric])
            and finite(right[training_seed][int(pid)][metric])
            for training_seed in training_seeds
        )
    ]
    if not metric_patient_ids:
        return {
            "delta": None,
            "ci95": [None, None],
            "ci95_low": None,
            "ci95_high": None,
            "per_seed_delta": {str(training_seed): None for training_seed in training_seeds},
            "patient_count": 0,
        }

    ids = np.asarray(metric_patient_ids, dtype=np.int64)
    observed_by_seed = []
    delta_vectors = []
    for training_seed in training_seeds:
        delta = np.asarray(
            [left[training_seed][int(pid)][metric] - right[training_seed][int(pid)][metric] for pid in ids],
            dtype=np.float64,
        )
        observed_by_seed.append(float(delta.mean()))
        delta_vectors.append(delta)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, len(ids), size=len(ids))
        samples[index] = np.mean([values[draw].mean() for values in delta_vectors])
    ci95_low = float(np.percentile(samples, 2.5))
    ci95_high = float(np.percentile(samples, 97.5))
    return {
        "delta": float(np.mean(observed_by_seed)),
        "ci95": [ci95_low, ci95_high],
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "per_seed_delta": {
            str(training_seed): value
            for training_seed, value in zip(training_seeds, observed_by_seed)
        },
        "patient_count": len(metric_patient_ids),
    }


def waiting_time_valid_patient_ids(rows, patient_ids) -> list[int]:
    reference = None
    reference_name = None
    for model_name, by_seed in rows.items():
        for training_seed, patient_rows in by_seed.items():
            current = {
                int(pid)
                for pid in patient_ids
                if int(patient_rows[int(pid)]["waiting_time_nll_valid"]) == 1
            }
            if reference is None:
                reference = current
                reference_name = f"{model_name}/seed{training_seed}"
            elif current != reference:
                raise ValueError(
                    "waiting_time_nll_valid is not model/seed invariant: "
                    f"{reference_name} != {model_name}/seed{training_seed}"
                )
    return sorted(reference or ())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_g_protocol(args.protocol, REPO_DIR)
    assert_training_allowed(protocol)
    assert_split_allowed(protocol, args.split)
    root = args.input_root or REPO_DIR / protocol["output_root"] / args.split
    out_dir = args.out_dir or REPO_DIR / protocol["output_root"] / f"{args.split}_assessment"
    model_names = [model["name"] for model in protocol["models"]]
    rows = {name: {} for name in model_names}
    patient_set = None
    for seed in protocol["seeds"]:
        for name in model_names:
            path = root / f"seed{seed}" / name / "patient_rows.json.gz"
            rows[name][seed] = read_rows(path)
            current = set(rows[name][seed])
            if patient_set is None:
                patient_set = current
            elif current != patient_set:
                raise ValueError(f"patient pairing mismatch: {path}")
    patient_ids = sorted(patient_set or ())
    if not patient_ids:
        raise ValueError("no paired Track G patients")
    waiting_time_valid_ids = waiting_time_valid_patient_ids(rows, patient_ids)

    statistics = protocol["statistics"]
    contrasts = [statistics["primary_contrast"], *statistics["external_contrasts"]]
    results = {}
    for contrast in contrasts:
        name = f"{contrast['left']}_minus_{contrast['right']}"
        results[name] = {}
        for metric in HEADLINE_METRICS:
            result = bootstrap_delta(
                rows[contrast["left"]],
                rows[contrast["right"]],
                patient_ids,
                metric,
                int(statistics["bootstrap_replicates"]),
                int(statistics["bootstrap_seed"]),
            )
            if metric == "waiting_time_nll" and result["patient_count"] != len(waiting_time_valid_ids):
                raise ValueError(
                    "waiting_time_nll finite paired set differs from the data-level valid set"
                )
            results[name][metric] = result
    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "split": args.split,
        "patient_count": len(patient_ids),
        "waiting_time_nll_data_valid_patient_count": len(waiting_time_valid_ids),
        "training_seeds": protocol["seeds"],
        "contrasts": results,
        "locked_test_read": protocol["locked_test_read"],
    }
    atomic_json(out_dir / "assessment.json", payload)
    atomic_json(out_dir / "status.json", {"status": "finished", "patient_count": len(patient_ids)})
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
