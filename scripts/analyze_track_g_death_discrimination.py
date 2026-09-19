"""Summarize rollout death discrimination without fitting or tuning a model."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(args.input_root.glob("seed*/*/patient_rows.json.gz")):
        model = path.parent.name
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                patient_index = int(row["patient_index"])
                item = predictions[model].setdefault(
                    patient_index,
                    {"label": int(row["actual_death"]), "probabilities": []},
                )
                if item["label"] != int(row["actual_death"]):
                    raise ValueError(f"inconsistent label for {model} patient {patient_index}")
                item["probabilities"].append(float(row["death_probability"]))

    summary = {}
    for model, patient_rows in sorted(predictions.items()):
        labels = np.asarray([row["label"] for row in patient_rows.values()], dtype=int)
        probabilities = np.asarray(
            [np.mean(row["probabilities"]) for row in patient_rows.values()], dtype=float
        )
        prevalence = float(labels.mean())
        brier = float(brier_score_loss(labels, probabilities))
        constant_brier = float(brier_score_loss(labels, np.full_like(probabilities, prevalence)))
        summary[model] = {
            "patient_count": int(labels.size),
            "seeds_per_patient": sorted({len(row["probabilities"]) for row in patient_rows.values()}),
            "death_prevalence": prevalence,
            "mean_probability": float(probabilities.mean()),
            "brier": brier,
            "constant_prevalence_brier": constant_brier,
            "brier_skill_score": float(1.0 - brier / constant_brier),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
