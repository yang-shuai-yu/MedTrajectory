from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_calibration_auc import load_model, load_split, resolve_vocab_csv  # noqa: E402
from semantic_delphi_ukb.paper_medical_auc import longitudinal_delong_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delphi-2M paper-aligned frozen longitudinal DeLong evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minimum-cases", type=int, default=25)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--max-patients", type=int, default=0)
    return parser


def load_outcomes(path: Path, token_ids: list[int], patient_count: int) -> np.ndarray:
    token_to_column = {token_id: index for index, token_id in enumerate(token_ids)}
    outcomes = np.zeros((patient_count, len(token_ids)), dtype=np.int8)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_index = int(row["row_index"])
            if row_index >= patient_count:
                continue
            column = token_to_column.get(int(row["token_id"]))
            if column is not None:
                outcomes[row_index, column] = 1
    return outcomes


def load_model_eligible(path: Path, patient_count: int) -> np.ndarray:
    eligible = np.zeros(patient_count, dtype=bool)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_index = int(row["row_index"])
            if row_index < patient_count:
                eligible[row_index] = bool(int(row["model_eligible"]))
    return eligible


def cutoff_landmark_positions(
    input_tokens: torch.Tensor,
    input_ages: torch.Tensor,
    followup_end_age_days: np.ndarray,
    model_eligible: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    cutoff = torch.tensor(
        np.asarray(followup_end_age_days).tolist(), dtype=input_ages.dtype, device=input_ages.device
    ).view(-1, 1)
    eligible = torch.tensor(
        np.asarray(model_eligible).tolist(), dtype=torch.bool, device=input_tokens.device
    )
    matches = (input_tokens == 1) & (torch.abs(input_ages - cutoff) <= 0.51)
    indices = torch.arange(input_tokens.size(1), device=input_tokens.device).view(1, -1).expand_as(input_tokens)
    positions = torch.where(matches, indices, torch.full_like(indices, -1)).max(dim=1).values
    keep = eligible & (positions >= 0)
    return keep, positions.clamp_min(0)


def build_cutoff_landmark_batch(
    patient_ids: np.ndarray,
    data: np.ndarray,
    p2i: np.ndarray,
    static: np.ndarray,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = np.zeros((len(patient_ids), block_size), dtype=np.int64)
    ages = np.full((len(patient_ids), block_size), -10000.0, dtype=np.float32)
    for batch_row, patient_id in enumerate(patient_ids):
        start, length = p2i[int(patient_id)]
        rows = data[int(start) : int(start) + int(length)][-block_size:]
        tokens[batch_row, -len(rows) :] = rows[:, 2].astype(np.int64) + 1
        ages[batch_row, -len(rows) :] = rows[:, 1].astype(np.float32)
    return (
        torch.tensor(tokens.tolist(), dtype=torch.long, device=device),
        torch.tensor(ages.tolist(), dtype=torch.float32, device=device),
        torch.tensor(static[patient_ids].tolist(), dtype=torch.float32, device=device),
    )


def token_metadata(vocab_csv: Path) -> dict[int, dict[str, str]]:
    with vocab_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["token_id"]): {
                "token_name": row.get("token_name", ""),
                "code_norm": row.get("code_norm", ""),
            }
            for row in csv.DictReader(handle)
        }


@torch.no_grad()
def infer_scores(model, data, p2i, static, token_ids, followup_end_ages, model_eligible, args):
    score_parts, patient_parts = [], []
    for start in range(0, len(p2i), args.batch_size):
        patient_ids = np.arange(start, min(start + args.batch_size, len(p2i)), dtype=np.int64)
        x, age, s = build_cutoff_landmark_batch(
            patient_ids,
            data,
            p2i,
            static,
            int(model.config.block_size),
            args.device,
        )
        logits, _, _, _, _ = model(x, age, s)
        keep, pos = cutoff_landmark_positions(
            x,
            age,
            followup_end_ages[patient_ids],
            model_eligible[patient_ids],
        )
        if not bool(keep.any()):
            continue
        batch = torch.arange(len(patient_ids), device=logits.device)[keep]
        selected = logits[batch, pos[keep]][:, token_ids]
        score_parts.append(selected.float().cpu().numpy())
        patient_parts.append(patient_ids[keep.cpu().numpy()])
    if not score_parts:
        raise RuntimeError("No model-eligible longitudinal patients")
    return np.concatenate(score_parts), np.concatenate(patient_parts)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = load_model(checkpoint, args.device)
    manifest = json.loads((args.data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    token_ids = [int(value) for value in manifest["longitudinal_eligible_disease_tokens"]]
    data, p2i, static = load_split(args.data_dir, "longitudinal", args.max_patients)
    followup_end_ages = np.load(args.data_dir / "longitudinal_followup_end_age_days.npy").astype(np.float32)
    model_eligible = load_model_eligible(args.data_dir / "longitudinal_patient_index.csv", len(p2i))
    if args.max_patients > 0:
        followup_end_ages = followup_end_ages[: args.max_patients]
    if len(followup_end_ages) != len(p2i):
        raise ValueError("longitudinal follow-up ages are not aligned with longitudinal.bin")
    outcomes = load_outcomes(args.data_dir / "longitudinal_outcomes.csv", token_ids, len(p2i))
    scores, patient_ids = infer_scores(
        model, data, p2i, static, token_ids, followup_end_ages, model_eligible, args
    )
    rows = longitudinal_delong_rows(scores, outcomes[patient_ids], token_ids, args.minimum_cases)
    metadata = token_metadata(resolve_vocab_csv(args.data_dir))
    for row in rows:
        row.update(metadata.get(int(row["disease_id"]), {}))
    summary = {
        "metric": "delphi2m_longitudinal_auc",
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "input_end": "2020-06-30",
        "gap": "2020-07-01..2021-06-30",
        "outcomes": "2021-07-01..2022-07-01",
        "minimum_cases": args.minimum_cases,
        "score_landmark": "explicit_no_event_token_at_2020-06-30_cutoff",
        "cohort_patients": len(p2i),
        "model_eligible_patients": int(len(patient_ids)),
        "eligible_tokens_configured": len(token_ids),
        "tokens_reported": len(rows),
        "mean_auc_delong": float(np.mean([row["auc_delong"] for row in rows])) if rows else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "longitudinal_auc_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
