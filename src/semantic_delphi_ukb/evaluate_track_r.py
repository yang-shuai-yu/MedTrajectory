"""Evaluate Track R neural models on one frozen fixed-position landmark manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.calibration_auc import build_horizon_case_control  # noqa: E402
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.evaluate_calibration_auc import build_official_left_batch  # noqa: E402
from semantic_delphi_ukb.evaluate_medical_control_tasks import _sanitize_json, _summary  # noqa: E402
from semantic_delphi_ukb.medbert_r_model import MedBERTR, MedBERTRConfig  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    STATIC_CONDITIONING_MODES,
    load_track_r_assets,
    load_track_r_static_features,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.track_r_contract import (  # noqa: E402
    load_track_r_protocol,
    medbert_attention_allowed,
    prepend_static_context,
)
from semantic_delphi_ukb.track_r_rows import write_json_gzip  # noqa: E402
from semantic_delphi_ukb.track_r_v2_2 import load_wavelength_manifest  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import load_followup_end_ages  # noqa: E402
from semantic_delphi_ukb.train_car_rope import load_split  # noqa: E402
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages, load_selected_disease_token_groups  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("carope", "medbert"), required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--landmark-manifest", type=Path, required=True)
    p.add_argument("--diseases-yaml", type=Path, default=None, help="override protocol diseases_yaml (e.g. expanded panel)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--include-static-prefix", choices=("true", "false"), required=True)
    p.add_argument("--static-conditioning", choices=STATIC_CONDITIONING_MODES, default="none")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument(
        "--residual-rope-alpha-override",
        type=float,
        default=None,
        help="same-checkpoint mechanism ablation; set learned residual alpha to a value in [0,1]",
    )
    return p


def load_model(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if args.family == "carope":
        config = checkpoint.get("model_args")
        if config is None:
            raise ValueError("CARoPE checkpoint is missing model_args")
        model = CARoPEHorizonMedTrajectory(CARoPEConfig(**config))
    else:
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("Med-BERT checkpoint is missing config")
        model = MedBERTR(MedBERTRConfig(**config))
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.family == "carope" and args.residual_rope_alpha_override is not None:
        model.set_residual_rope_alpha_override(args.residual_rope_alpha_override)
    return model.to(args.device).eval()


def validate_v2_2_wavelength_manifest(protocol: dict) -> Optional[dict]:
    if protocol["protocol_id"] != "track_r_v2_2":
        return None
    contract = protocol["wavelength_contract"]
    if not contract.get("expected_sha256"):
        raise ValueError("v2.2 evaluation requires wavelength_contract.expected_sha256")
    path = Path(contract["manifest"])
    if not path.is_absolute():
        path = ROOT / path
    return load_wavelength_manifest(
        path,
        expected_sha256=contract.get("expected_sha256"),
        scale_factor_bounds=contract["scale_factor_bounds"],
        require_log_scale_match=bool(contract["require_manifest_log_scale_match"]),
    )


def landmark_index(payload: dict) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for entry in payload.get("landmarks", []):
        if "position" not in entry:
            raise ValueError("Track R requires fixed-position landmarks")
        grouped[int(entry["patient_index"])].append(entry)
    return grouped


@torch.no_grad()
def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")
    protocol = load_track_r_protocol(args.protocol)
    wavelength_manifest = validate_v2_2_wavelength_manifest(protocol)
    manifest = validate_track_r_data_manifest(args.data_dir, protocol)
    landmark_payload = json.loads(args.landmark_manifest.read_text(encoding="utf-8-sig"))
    if landmark_payload.get("split") != args.split:
        raise ValueError("landmark split mismatch")
    grouped = landmark_index(landmark_payload)
    include_static = args.include_static_prefix == "true"
    if args.family == "medbert" and not include_static:
        raise ValueError("registered Med-BERT-R variants require the static prefix")
    if args.static_conditioning != "none" and include_static:
        raise ValueError("static residual conditioning cannot be combined with static prefix tokens")
    if args.family != "carope" and args.static_conditioning != "none":
        raise ValueError("static residual conditioning is registered only for CARoPE diagnostics")
    model = load_model(args)
    if args.family == "carope" and model.config.age_rope_variant == "additive_v2_2":
        checkpoint_wavelengths = tuple(float(value) for value in model.config.rope_wavelengths_days)
        manifest_wavelengths = tuple(float(value) for value in wavelength_manifest["wavelengths_days_head_major"])
        if checkpoint_wavelengths != manifest_wavelengths:
            raise ValueError("checkpoint RoPE wavelengths differ from the frozen manifest")
    data, p2i, static = load_split(args.data_dir, args.split, 0)
    prefix_ids, anchor_ages = load_track_r_assets(args.data_dir, args.split)
    conditioning = load_track_r_static_features(args.data_dir, args.split, args.static_conditioning)
    if args.family == "carope" and int(model.config.static_dim) != int(conditioning.shape[1]):
        raise ValueError(
            f"checkpoint static_dim={model.config.static_dim} does not match "
            f"conditioning width={conditioning.shape[1]}"
        )
    patient_ids, dynamic_batch = build_official_left_batch(
        data, p2i, static, int(protocol["dynamic_context_length"]), no_event_token_rate=5
    )
    x_dynamic, age_dynamic, y_dynamic, target_age_dynamic, _ = dynamic_batch
    diseases, token_groups = load_selected_disease_token_groups(args.diseases_yaml or (ROOT / protocol["diseases_yaml"]), args.data_dir)
    disease_ages, observed_last = build_patient_disease_ages(data, p2i, token_groups, int(manifest["vocab_size"]))
    followup = load_followup_end_ages(args.data_dir, args.split, observed_last, 0)
    horizons = tuple(float(value) for value in protocol["horizons_years"])
    id_to_key = {int(value): key for key, value in manifest["static_token_ids"].items()}
    bos_token_id = int(manifest["dynamic_bos_token_id"])
    rows = []

    for start in range(0, len(patient_ids), args.batch_size):
        stop = min(start + args.batch_size, len(patient_ids))
        batch_patient_ids = patient_ids[start:stop].astype(np.int64)
        prefix = torch.as_tensor(prefix_ids[batch_patient_ids], dtype=torch.long)
        anchors = torch.as_tensor(anchor_ages[batch_patient_ids], dtype=torch.float32)
        if include_static:
            x, age, y, target_age, static_mask, bos_mask, _ = prepend_static_context(
                x_dynamic[start:stop], age_dynamic[start:stop], y_dynamic[start:stop],
                target_age_dynamic[start:stop], prefix, anchors, bos_token_id,
            )
            position_offset = prefix.size(1) + 1
        else:
            x, age, y, target_age, static_mask, bos_mask, _ = prepend_static_context(
                x_dynamic[start:stop], age_dynamic[start:stop], y_dynamic[start:stop],
                target_age_dynamic[start:stop], prefix[:, :0], torch.zeros_like(anchors), bos_token_id,
            )
            position_offset = 1
        x, age, y, target_age, static_mask, bos_mask = [
            value.to(args.device) for value in (x, age, y, target_age, static_mask, bos_mask)
        ]
        static_features = None
        static_feature_mask = None
        if args.static_conditioning != "none":
            static_features = torch.as_tensor(conditioning[batch_patient_ids], dtype=torch.float32, device=args.device)
            if args.static_conditioning == "categorical-residual":
                static_feature_mask = (x > 0) & (age >= anchors.to(args.device)[:, None])

        selected = []
        for local_row, patient_id in enumerate(batch_patient_ids):
            for entry in grouped.get(int(patient_id), []):
                position = int(entry["position"]) + position_offset
                if position < position_offset or position >= x.size(1):
                    raise ValueError(f"frozen position out of range for patient_index={patient_id}")
                if not math.isclose(float(age[local_row, position].cpu()), float(entry["target_age_days"]), abs_tol=0.25):
                    raise ValueError(f"frozen position age mismatch for patient_index={patient_id}")
                selected.append((local_row, int(patient_id), position, entry))
        if not selected:
            continue

        if args.family == "carope":
            risk = torch.sigmoid(
                model(
                    x,
                    age,
                    static_features,
                    static_token_mask=static_mask,
                    static_feature_mask=static_feature_mask,
                    bos_token_mask=bos_mask,
                )[4]
            )
            selected_scores = torch.stack([risk[row, position] for row, _, position, _ in selected]).cpu().numpy()
        else:
            model_inputs, model_ages, model_static, positions = [], [], [], []
            for local_row, _, position, _ in selected:
                tokens = x[local_row].clone()
                future = torch.arange(tokens.size(0), device=args.device) > position
                tokens[future] = 0
                static_row = static_mask[local_row] & ~future
                model_inputs.append(tokens)
                model_ages.append(age[local_row])
                model_static.append(static_row)
                positions.append(position)
            tokens = torch.stack(model_inputs)
            ages = torch.stack(model_ages)
            static_rows = torch.stack(model_static)
            positions_tensor = torch.as_tensor(positions, device=args.device)
            allowed = medbert_attention_allowed(tokens, ages, static_rows)
            selected_scores = torch.sigmoid(
                model(tokens, ages, static_rows, attention_allowed=allowed, prediction_positions=positions_tensor)["risk_logits"]
            ).cpu().numpy()

        for selected_idx, (_local_row, patient_id, position, entry) in enumerate(selected):
            prediction_age = float(entry["target_age_days"])
            sex_key = next(id_to_key[int(value)] for value in prefix_ids[patient_id] if id_to_key[int(value)].startswith("static:sex:"))
            sex = sex_key.rsplit(":", 1)[-1]
            for horizon_idx, horizon in enumerate(horizons):
                for disease_idx, disease in enumerate(diseases):
                    labels, eligible = build_horizon_case_control(
                        np.asarray([patient_id]), np.asarray([prediction_age]), disease_ages,
                        followup, horizon, disease_idx,
                    )
                    if not bool(eligible[0]):
                        continue
                    rows.append({
                        "model": args.model_name,
                        "patient_index": patient_id,
                        "horizon_years": horizon,
                        "disease_id": disease.disease_id,
                        "label": int(labels[0]),
                        "score": float(selected_scores[selected_idx, horizon_idx, disease_idx]),
                        "score_is_probability": True,
                        "sex": sex,
                        "prediction_age_days": prediction_age,
                        "position_age_days": float(age[_local_row, position].cpu()),
                        "age_start_years": float(entry["age_start_years"]),
                    })

    summary = _summary(rows, args.model_name, args.calibration_bins)
    payload = {
        "protocol": protocol["protocol_id"],
        "split": args.split,
        "family": args.family,
        "model": args.model_name,
        "include_static_prefix": include_static,
        "static_conditioning": args.static_conditioning,
        "static_fusion_stage": getattr(model.config, "static_fusion_stage", None),
        "residual_rope_mode": getattr(model.config, "residual_rope_mode", "none"),
        "residual_rope_alpha_override": args.residual_rope_alpha_override,
        "residual_rope_diagnostics": (
            model.residual_rope_diagnostics()
            if args.family == "carope" and getattr(model.config, "residual_rope_mode", "none") != "none"
            else None
        ),
        "medbert_future_context_policy": "truncate_after_each_frozen_landmark" if args.family == "medbert" else None,
        "static_temporal_visibility": protocol["static_prefix"]["age_anchor"]["temporal_visibility"],
        "landmark_manifest": str(args.landmark_manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "patient_rows": len(rows),
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True)
    rows_path = write_json_gzip(args.out_dir / "rows.json.gz", rows)
    payload["validation_rows"] = str(rows_path.resolve())
    payload["validation_rows_format"] = "json.gz"
    (args.out_dir / "summary.json").write_text(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_sanitize_json(summary["macro"]), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
