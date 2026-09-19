"""Audit frozen landmark coverage across profile-specific test histories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.calibration_auc import select_shared_age_landmarks  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import build_official_left_batch  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import load_split  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True, help="name=data_dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--common-out", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    shared = defaultdict(list)
    for entry in manifest["landmarks"]:
        shared[int(entry["patient_index"])].append(entry)
    expected = int(manifest["landmark_count"])
    max_targets = max((len(entries) for entries in shared.values()), default=0)
    fixed_padding = np.full((int(manifest.get("patient_count", 0)), max_targets), -10000.0, dtype=np.float32)
    for patient_id, entries in shared.items():
        for index, entry in enumerate(entries):
            fixed_padding[int(patient_id), index] = float(entry["target_age_days"])
    profiles = {}
    errors = []
    resolved_keys = []
    for spec in args.profile:
        name, raw_dir = spec.split("=", 1)
        data_dir = Path(raw_dir)
        data, p2i, static = load_split(data_dir, manifest["split"], max_patients=0)
        patient_ids, batch = build_official_left_batch(data, p2i, static, int(manifest["block_size"]), no_event_token_rate=5, fixed_padding_ages=fixed_padding)
        selected = select_shared_age_landmarks(batch[0].numpy(), batch[1].numpy(), patient_ids, shared)
        resolved = len(selected["patient_ids"])
        keys = {
            (int(patient_id), float(age_start))
            for patient_id, age_start in zip(selected["patient_ids"], selected["age_start_years"])
        }
        resolved_keys.append(keys)
        lags = selected["prediction_age_days"] - selected["position_age_days"]
        profile = {
            "data_dir": str(data_dir.resolve()),
            "patient_count": len(patient_ids),
            "expected_landmarks": expected,
            "resolved_landmarks": resolved,
            "missing_landmarks": expected - resolved,
            "max_position_lag_days": float(lags.max()) if resolved else None,
            "positions_after_target": int((lags < 0).sum()),
        }
        profiles[name] = profile
        if resolved != expected:
            errors.append(f"missing_landmarks:{name}:{expected - resolved}")
        if profile["positions_after_target"]:
            errors.append(f"positions_after_target:{name}:{profile['positions_after_target']}")
    payload = {"protocol": manifest["protocol"], "manifest": str(args.manifest.resolve()), "profiles": profiles, "errors": errors, "ok": not errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.common_out is not None and resolved_keys:
        common_keys = set.intersection(*resolved_keys)
        common_entries = [
            entry for entry in manifest["landmarks"]
            if (int(entry["patient_index"]), float(entry["age_start_years"])) in common_keys
        ]
        common_manifest = dict(manifest)
        common_manifest["protocol"] = "paper_medical_control_shared_landmarks_intersection_v1"
        common_manifest["source_manifest"] = str(args.manifest.resolve())
        common_manifest["landmarks"] = common_entries
        common_manifest["landmark_count"] = len(common_entries)
        common_manifest["profile_count"] = len(resolved_keys)
        args.common_out.parent.mkdir(parents=True, exist_ok=True)
        args.common_out.write_text(json.dumps(common_manifest, indent=2), encoding="utf-8")
        payload["common_manifest"] = {"path": str(args.common_out.resolve()), "landmark_count": len(common_entries)}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
