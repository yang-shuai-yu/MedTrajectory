"""Build one leakage-controlled landmark feature artifact for Track R baselines."""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_r_batch import load_track_r_assets, validate_track_r_data_manifest  # noqa: E402
from semantic_delphi_ukb.calibration_auc import horizon_case_control_at_age  # noqa: E402
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402
from semantic_delphi_ukb.train_car_rope import load_split  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import load_followup_end_ages  # noqa: E402
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages, load_selected_disease_token_groups  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    p.add_argument("--landmark-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    manifest = validate_track_r_data_manifest(args.data_dir, protocol)
    landmarks = json.loads(args.landmark_manifest.read_text(encoding="utf-8-sig"))
    if landmarks.get("split") != args.split:
        raise ValueError("landmark split mismatch")
    data, p2i, static = load_split(args.data_dir, args.split, 0)
    prefix_ids, anchor_ages = load_track_r_assets(args.data_dir, args.split)
    diseases, token_groups = load_selected_disease_token_groups(ROOT / protocol["diseases_yaml"], args.data_dir)
    disease_ages, observed_last = build_patient_disease_ages(data, p2i, token_groups, int(manifest["vocab_size"]))
    followup = load_followup_end_ages(args.data_dir, args.split, observed_last, 0)
    horizons = np.asarray(protocol["horizons_years"], dtype=np.float32)
    static_token_ids = manifest["static_token_ids"]
    ordered_static_keys = sorted(static_token_ids, key=static_token_ids.get)
    model_static_keys = [key for key in ordered_static_keys if not key.endswith(":missing")]
    static_columns = {int(static_token_ids[key]): index for index, key in enumerate(model_static_keys)}
    feature_names = ["landmark_age_years", "static_available"] + [f"onehot:{key}" for key in model_static_keys]
    for disease in diseases:
        feature_names.extend((f"history_ever:{disease.disease_id}", f"history_count:{disease.disease_id}", f"history_recency_years:{disease.disease_id}"))

    entries = landmarks.get("landmarks", [])
    n = len(entries)
    x = np.zeros((n, len(feature_names)), dtype=np.float32)
    labels = np.zeros((n, len(horizons), len(diseases)), dtype=np.float32)
    label_mask = np.zeros_like(labels)
    survival_duration = np.zeros((n, len(diseases)), dtype=np.float32)
    survival_event = np.zeros_like(survival_duration)
    patient_index = np.zeros(n, dtype=np.int64)
    prediction_age_days = np.zeros(n, dtype=np.float32)
    age_start_years = np.zeros(n, dtype=np.float32)
    sex = np.empty(n, dtype="U16")
    static_available_rows = 0

    id_to_key = {int(value): key for key, value in static_token_ids.items()}
    for row_idx, entry in enumerate(entries):
        pid = int(entry["patient_index"])
        if pid < 0 or pid >= len(p2i):
            raise ValueError(f"landmark patient_index is out of range: {pid}")
        current_age = float(entry["target_age_days"])
        patient_index[row_idx] = pid
        prediction_age_days[row_idx] = current_age
        age_start_years[row_idx] = float(entry["age_start_years"])
        x[row_idx, 0] = current_age / 365.25
        static_is_available = current_age >= float(anchor_ages[pid])
        x[row_idx, 1] = float(static_is_available)
        static_available_rows += int(static_is_available)
        keys = [id_to_key[int(value)] for value in prefix_ids[pid]]
        sex_key = next(key for key in keys if key.startswith("static:sex:"))
        sex[row_idx] = sex_key.rsplit(":", 1)[-1]
        if static_is_available:
            for token in prefix_ids[pid]:
                if int(token) in static_columns:
                    x[row_idx, 2 + static_columns[int(token)]] = 1.0
        feature_offset = 2 + len(model_static_keys)
        for disease_idx, ages in enumerate(disease_ages[pid]):
            before = bisect.bisect_right(ages, current_age)
            if before:
                x[row_idx, feature_offset + 3 * disease_idx] = 1.0
                x[row_idx, feature_offset + 3 * disease_idx + 1] = float(before)
                x[row_idx, feature_offset + 3 * disease_idx + 2] = (current_age - float(ages[before - 1])) / 365.25
            next_age = float(ages[before]) if before < len(ages) else float("inf")
            censor_age = float(followup[pid])
            duration_days = max(1.0, min(next_age, censor_age) - current_age)
            event = bool(next_age <= censor_age)
            survival_duration[row_idx, disease_idx] = duration_days / 365.25
            survival_event[row_idx, disease_idx] = float(event)
            for horizon_idx, horizon in enumerate(horizons):
                label, eligible = horizon_case_control_at_age(current_age, ages, censor_age, float(horizon))
                labels[row_idx, horizon_idx, disease_idx] = float(label)
                label_mask[row_idx, horizon_idx, disease_idx] = float(eligible)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x,
        labels=labels,
        label_mask=label_mask,
        survival_duration=survival_duration,
        survival_event=survival_event,
        patient_index=patient_index,
        prediction_age_days=prediction_age_days,
        age_start_years=age_start_years,
        sex=sex,
    )
    schema = {
        "protocol_id": protocol["protocol_id"],
        "split": args.split,
        "source_landmark_manifest": str(args.landmark_manifest.resolve()),
        "feature_names": feature_names,
        "categorical_reference": "missing",
        "disease_ids": [disease.disease_id for disease in diseases],
        "horizons_years": horizons.tolist(),
        "rows": n,
        "static_available_rows": static_available_rows,
        "static_unavailable_rows": n - static_available_rows,
        "redundancy_policy": protocol["redundancy_policy"],
    }
    args.output.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
