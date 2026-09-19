from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from semantic_delphi_ukb.multitype_batch import get_batch
from semantic_delphi_ukb.selected_disease_demo import binary_auc, parse_selected_diseases, top_decile_stats, write_csv
from semantic_delphi_ukb.tte_model import TTEConfig, TTEMultitaskDelphi
from semantic_delphi_ukb.tte_selected_utils import tte_checkpoint_meta, tte_model_spec
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages, build_tte_batch, load_selected_disease_token_groups
from utils import get_p2i


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate selected diseases with the TTE head hazard scores.")
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "tests" / "output" / "selected_disease_tte_head_direct")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--horizons", type=str, default="5,10")
    return parser


def parse_horizons(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("horizons must contain at least one positive value.")
    return [value for value in values if value > 0]


def load_full_tte_model(split: str, device: str):
    spec = tte_model_spec()
    checkpoint = torch.load(spec.ckpt_path, map_location=device, weights_only=False)
    conf = TTEConfig(**checkpoint["model_args"])
    model = TTEMultitaskDelphi(conf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data = np.memmap(spec.data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static_matrix = np.load(spec.data_dir / f"{split}_static.npy").astype(np.float32)
    return spec, model, data, get_p2i(data), static_matrix, checkpoint


def harrell_c_index(scores: np.ndarray, events: np.ndarray, durations: np.ndarray) -> float:
    if len(scores) == 0 or int(events.sum()) == 0:
        return float("nan")

    score_values = sorted(set(float(score) for score in scores))
    bit = [0] * (len(score_values) + 2)

    def add(index: int, value: int) -> None:
        while index < len(bit):
            bit[index] += value
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += bit[index]
            index -= index & -index
        return total

    order = np.argsort(-durations, kind="mergesort")
    comparable = 0
    concordant = 0.0
    processed = 0
    start = 0
    while start < len(order):
        stop = start + 1
        duration = durations[order[start]]
        while stop < len(order) and durations[order[stop]] == duration:
            stop += 1

        for row_index in order[start:stop]:
            if events[row_index] <= 0:
                continue
            rank = bisect.bisect_left(score_values, float(scores[row_index])) + 1
            lower = prefix(rank - 1)
            equal = prefix(rank) - lower
            comparable += processed
            concordant += lower + 0.5 * equal

        for row_index in order[start:stop]:
            rank = bisect.bisect_left(score_values, float(scores[row_index])) + 1
            add(rank, 1)
            processed += 1
        start = stop

    return float(concordant / comparable) if comparable else float("nan")


def collect_scores(
    model: TTEMultitaskDelphi,
    data: np.ndarray,
    p2i: np.ndarray,
    static_matrix: np.ndarray,
    token_groups: Sequence[Sequence[int]],
    horizons: Sequence[float],
    batch_size: int,
    max_patients: int,
    no_event_token_rate: int,
    device: str,
):
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(
        data=data,
        p2i=p2i,
        token_groups=token_groups,
        vocab_size=int(model.config.vocab_size),
    )
    num_patients = len(p2i) if max_patients <= 0 else min(len(p2i), max_patients)
    buckets = {
        float(horizon): [
            {"scores": [], "events": [], "durations": [], "masks": []}
            for _ in token_groups
        ]
        for horizon in horizons
    }

    with torch.no_grad():
        for start in range(0, num_patients, batch_size):
            stop = min(start + batch_size, num_patients)
            ix = list(range(start, stop))
            x, ages, y, target_ages, static_features = get_batch(
                ix,
                data,
                p2i,
                static_matrix,
                block_size=int(model.config.block_size),
                device=device,
                no_event_token_rate=no_event_token_rate,
                padding="regular",
                cut_batch=True,
            )
            _, _, _, tte_logits = model(
                x,
                ages,
                static_features,
                y,
                target_ages,
                validation_loss_mode=True,
            )
            hazard = F.softplus(tte_logits).detach().cpu().numpy().astype(np.float64)

            for horizon in horizons:
                events, durations, mask = build_tte_batch(
                    ix,
                    ages,
                    x > 1,
                    patient_disease_ages,
                    patient_last_ages,
                    float(horizon),
                    device,
                )
                events_np = events.detach().cpu().numpy().astype(np.float64)
                durations_np = durations.detach().cpu().numpy().astype(np.float64)
                mask_np = mask.detach().cpu().numpy().astype(bool)
                for disease_idx in range(len(token_groups)):
                    valid = mask_np[..., disease_idx]
                    if not valid.any():
                        continue
                    bucket = buckets[float(horizon)][disease_idx]
                    bucket["scores"].append(hazard[..., disease_idx][valid])
                    bucket["events"].append(events_np[..., disease_idx][valid])
                    bucket["durations"].append(durations_np[..., disease_idx][valid])
                    bucket["masks"].append(valid.reshape(-1))

    return buckets


def summarize(diseases, token_groups, buckets) -> list[dict]:
    rows = []
    for horizon, per_disease in buckets.items():
        for disease_idx, disease in enumerate(diseases):
            bucket = per_disease[disease_idx]
            if bucket["scores"]:
                scores = np.concatenate(bucket["scores"])
                events = np.concatenate(bucket["events"]).astype(np.int8)
                durations = np.concatenate(bucket["durations"])
            else:
                scores = np.asarray([], dtype=np.float64)
                events = np.asarray([], dtype=np.int8)
                durations = np.asarray([], dtype=np.float64)
            positives = int(events.sum())
            total = int(events.size)
            negatives = total - positives
            auc = binary_auc(scores, events) if positives > 0 and negatives > 0 else float("nan")
            top_capture, top_rate, lift = top_decile_stats(scores, events)
            rows.append(
                {
                    "horizon_years": horizon,
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "category": disease.category,
                    "icd10": ";".join(disease.ranges),
                    "matched_tokens": len(token_groups[disease_idx]),
                    "prediction_moments": total,
                    "positives": positives,
                    "negatives": negatives,
                    "baseline_event_rate": positives / total if total else float("nan"),
                    "auc": auc,
                    "c_index": harrell_c_index(scores, events, durations),
                    "top_decile_capture": top_capture,
                    "top_decile_event_rate": top_rate,
                    "top_decile_lift": lift,
                    "mean_hazard": float(scores.mean()) if scores.size else float("nan"),
                }
            )
    return rows


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "# TTE Head Direct Selected-Disease Evaluation",
        "",
        "Scores are direct `tte_head` hazards, not next-token logits.",
        "",
        "| Horizon | Disease | Positives | AUC | C-index | Top-decile capture |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {horizon:g} | {name} | {pos} | {auc:.4f} | {cindex:.4f} | {cap:.4f} |".format(
                horizon=float(row["horizon_years"]),
                name=row["name"],
                pos=row["positives"],
                auc=float(row["auc"]),
                cindex=float(row["c_index"]),
                cap=float(row["top_decile_capture"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    diseases = parse_selected_diseases(args.diseases_yaml)
    _, token_groups = load_selected_disease_token_groups(args.diseases_yaml)
    _, model, data, p2i, static_matrix, checkpoint = load_full_tte_model(args.split, args.device)
    no_event_token_rate = int(checkpoint.get("config", {}).get("no_event_token_rate", 5))

    buckets = collect_scores(
        model=model,
        data=data,
        p2i=p2i,
        static_matrix=static_matrix,
        token_groups=token_groups,
        horizons=horizons,
        batch_size=args.batch_size,
        max_patients=args.max_patients,
        no_event_token_rate=no_event_token_rate,
        device=args.device,
    )
    rows = summarize(diseases, token_groups, buckets)
    write_csv(args.output_dir / "tte_head_metrics.csv", rows)
    write_summary(args.output_dir / "tte_head_summary.md", rows)
    (args.output_dir / "tte_head_details.json").write_text(
        json.dumps(
            {
                "checkpoint": tte_checkpoint_meta(),
                "split": args.split,
                "horizons": horizons,
                "max_patients": args.max_patients,
                "metrics": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
