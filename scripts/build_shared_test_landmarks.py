"""Build a frozen test landmark manifest from one canonical profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.calibration_auc import select_age_landmarks, validate_age_groups  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import build_official_left_batch  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import load_split  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--age-groups", default="50,55,60,65,70,75")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    age_groups, _ = validate_age_groups([float(value) for value in args.age_groups.split(",") if value.strip()])
    data, p2i, static = load_split(args.data_dir, args.split, args.max_patients)
    patient_ids, batch = build_official_left_batch(data, p2i, static, args.block_size, no_event_token_rate=5)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 9173]))
    landmarks = select_age_landmarks(batch[0].numpy(), batch[1].numpy(), patient_ids, age_groups, rng)
    entries = [
        {
            "patient_index": int(patient_id),
            "age_start_years": float(age_start),
            "target_age_days": float(target_age),
            "position": int(position),
        }
        for patient_id, age_start, target_age, position in zip(
            landmarks["patient_ids"], landmarks["age_start_years"],
            landmarks["prediction_age_days"], landmarks["positions"],
        )
    ]
    payload = {
        "protocol": "paper_medical_control_shared_landmarks_v1",
        "split": args.split,
        "source_data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "block_size": args.block_size,
        "age_groups_years": age_groups.tolist(),
        "patient_count": len(patient_ids),
        "landmark_count": len(entries),
        "landmarks": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "patient_count": len(patient_ids), "landmark_count": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
