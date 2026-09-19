"""Evaluate horizon controls with the Delphi medical age/sex protocol.

This evaluator keeps the same patient x disease x horizon rows for every
model.  Unlike ``evaluate_horizon_control_tasks.py`` it samples one landmark
per patient and five-year age bin, stratifies by sex, and applies the prepared
follow-up end ages for censor-aware controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.calibration_auc import (  # noqa: E402
    DAYS_PER_YEAR,
    age_stratified_delong_rows,
    build_horizon_case_control,
    select_age_landmarks,
    select_shared_age_landmarks,
    validate_age_groups,
)
from semantic_delphi_ukb.evaluate_horizon_control_tasks import (  # noqa: E402
    captured_hidden,
    disease_scores_from_lm,
    load_model,
)
from semantic_delphi_ukb.evaluate_calibration_auc import build_official_left_batch  # noqa: E402
from semantic_delphi_ukb.horizon_control_metrics import summarize_scores  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    build_patient_disease_ages,
    load_followup_end_ages,
    load_selected_disease_token_groups,
    load_split,
    parse_horizons,
    resolve_diseases_yaml,
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["lm", "explicit", "survival", "linear"], required=False)
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, required=False)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--age-groups", default="50,55,60,65,70,75")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--landmark-manifest", type=Path, default=None)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--lm-aggregation", choices=["logsumexp", "max"], default="logsumexp")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _summary(rows: list[dict], mode: str, bins: int) -> dict[str, object]:
    grouped: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["sex"], row["disease_id"], float(row["horizon_years"]), float(row["age_start_years"]))].append(row)
    cells: list[dict] = []
    probability_scores = mode != "lm"
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        labels = np.asarray([int(row["label"]) for row in group], dtype=np.int8)
        scores = np.asarray([float(row["score"]) for row in group], dtype=np.float64)
        stats = summarize_scores(scores, labels, bins=bins, probability_scores=probability_scores)
        cells.append({"sex": key[0], "disease_id": key[1], "horizon_years": key[2], "age_start_years": key[3], **stats})
    valid = [cell for cell in cells if np.isfinite(cell["auc"])]
    macro = {name: float(np.mean([cell[name] for cell in valid])) if valid else float("nan") for name in ("auc", "auprc", "brier", "ece")}
    return {"cells": cells, "cell_count": len(cells), "valid_auc_cells": len(valid), "macro": macro}


def _sanitize_json(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    for name in ("mode", "checkpoint", "data_dir", "out_dir"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required unless --self-test is used")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in args.out_dir.iterdir() if path.name != "stdout.log"]
    if existing:
        raise FileExistsError(f"evaluation output directory is not empty: {args.out_dir}")
    horizons = parse_horizons(args.horizons)
    age_groups, _ = validate_age_groups([float(value) for value in args.age_groups.split(",") if value.strip()])
    data, p2i, static = load_split(args.data_dir, args.split, args.max_patients)
    diseases, token_groups = load_selected_disease_token_groups(resolve_diseases_yaml(args.diseases_yaml), args.data_dir)
    manifest = json.loads((args.data_dir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    vocab_size = int(manifest.get("vocab_size", int(data[:, 2].max()) + 2))
    patient_disease_ages, observed_last = build_patient_disease_ages(data, p2i, token_groups, vocab_size)
    followup_end = load_followup_end_ages(args.data_dir, args.split, observed_last, args.max_patients)
    shared_landmarks = None
    landmark_manifest = None
    if args.landmark_manifest is not None:
        landmark_manifest = json.loads(args.landmark_manifest.read_text(encoding="utf-8-sig"))
        if landmark_manifest.get("split") != args.split:
            raise ValueError("shared landmark manifest split does not match evaluator split")
        manifest_groups = [float(value) for value in landmark_manifest.get("age_groups_years", [])]
        if manifest_groups != age_groups.tolist():
            raise ValueError("shared landmark manifest age groups do not match evaluator")
        shared_landmarks = defaultdict(list)
        for entry in landmark_manifest.get("landmarks", []):
            shared_landmarks[int(entry["patient_index"])].append(entry)
    model, checkpoint = load_model(args.mode, args.checkpoint, args.device)
    probe = None
    if args.mode == "linear":
        if args.probe_checkpoint is None:
            raise ValueError("--probe-checkpoint is required for --mode linear")
        payload = torch.load(args.probe_checkpoint, map_location=args.device, weights_only=False)
        if payload.get("horizons") != horizons or payload.get("disease_ids") != [d.disease_id for d in diseases]:
            raise ValueError("linear probe checkpoint does not match diseases/horizons")
        probe = torch.nn.Linear(int(payload["hidden_dim"]), len(horizons) * len(diseases)).to(args.device)
        probe.load_state_dict(payload["state_dict"], strict=True)
        probe.eval()

    # Match the official medical-AUC evaluator's landmark stream exactly.
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 9173]))
    rows: list[dict] = []
    patient_count = len(p2i)
    patient_ids, full_batch = build_official_left_batch(
        data, p2i, static, args.block_size, no_event_token_rate=5
    )
    for start in range(0, patient_count, args.batch_size):
        stop = min(start + args.batch_size, patient_count)
        x, age, y, _target_age, static_batch = [tensor[start:stop].to(args.device) for tensor in full_batch]
        ix = patient_ids[start:stop]
        if shared_landmarks is None:
            landmarks = select_age_landmarks(
                x.detach().cpu().numpy(), age.detach().cpu().numpy(), ix, age_groups, rng
            )
        else:
            landmarks = select_shared_age_landmarks(
                x.detach().cpu().numpy(), age.detach().cpu().numpy(), ix, shared_landmarks
            )
        if len(landmarks["row_indices"]) == 0:
            continue
        if args.mode == "lm":
            logits = model(x, age, static_batch)[0]
            position_scores = disease_scores_from_lm(logits, token_groups, args.lm_aggregation)
            position_scores = position_scores[:, :, None, :].expand(-1, -1, len(horizons), -1)
        elif args.mode == "explicit":
            position_scores = torch.sigmoid(model(x, age, static_batch)[4])
        elif args.mode == "survival":
            survival_logits = model(x, age, static_batch)[4]
            position_scores = model.horizon_risk_from_survival(survival_logits, horizons)
        else:
            hidden = captured_hidden(model, x, age, static_batch)
            position_scores = torch.sigmoid(probe(hidden).view(x.size(0), x.size(1), len(horizons), len(diseases)))
        selected_scores = position_scores[
            torch.as_tensor(landmarks["row_indices"], device=args.device),
            torch.as_tensor(landmarks["positions"], device=args.device),
        ].detach().cpu().numpy()
        landmark_patients = landmarks["patient_ids"].astype(np.int64)
        sex_values = static[landmark_patients, 0]
        for horizon_idx, horizon in enumerate(horizons):
            for disease_idx, disease in enumerate(diseases):
                labels, eligible = build_horizon_case_control(
                    landmark_patients, landmarks["prediction_age_days"], patient_disease_ages,
                    followup_end, horizon, disease_idx,
                )
                for row_idx in np.flatnonzero(eligible):
                    rows.append({
                        "model": args.mode,
                        "patient_index": int(landmark_patients[row_idx]),
                        "horizon_years": float(horizon),
                        "disease_id": disease.disease_id,
                        "label": int(labels[row_idx]),
                        "score": float(selected_scores[row_idx, horizon_idx, disease_idx]),
                        "score_is_probability": bool(args.mode != "lm"),
                        "sex": (
                            "female" if int(sex_values[row_idx]) == 0
                            else "male" if int(sex_values[row_idx]) == 1
                            else "missing"
                        ),
                        "prediction_age_days": float(landmarks["prediction_age_days"][row_idx]),
                        "position_age_days": float(landmarks.get("position_age_days", landmarks["prediction_age_days"])[row_idx]),
                        "age_start_years": float(landmarks["age_start_years"][row_idx]),
                    })

    summary = _summary(rows, args.mode, args.calibration_bins)
    payload = {
        "protocol": "paper_medical_control_v1",
        "protocol_signature": {
            "age_groups_years": age_groups.tolist(),
            "age_bin_width_years": 5.0,
            "sex_stratified": True,
            "case_control": "paper_censor_aware_v1",
            "followup_source": f"{args.split}_followup_end_age_days.npy",
            "landmark_selection": "one_random_valid_position_per_patient_age_bin",
            "left_window_padding": "random",
            "shared_landmark_manifest": str(args.landmark_manifest.resolve()) if args.landmark_manifest else None,
        },
        "split": args.split,
        "mode": args.mode,
        "data_dir": str(args.data_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "horizons_years": horizons,
        "age_groups": age_groups.tolist(),
        "censor_aware": True,
        "patient_rows": len(rows),
        "summary": summary,
    }
    (args.out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps(_sanitize_json(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return payload


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        groups, step = validate_age_groups([50, 55, 60, 65, 70, 75])
        assert step == 5.0 and len(groups) == 6
        print(json.dumps({"self_test": True, "protocol": "paper_medical_control_v1"}))
        return 0
    payload = evaluate(args)
    print(json.dumps(_sanitize_json(payload["summary"]["macro"]), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
