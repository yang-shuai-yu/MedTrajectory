"""Generation evaluation for CARoPE models trained on the simple MIMIC multitype format.

This is a lightweight, protocol-free rollout benchmark (no Track R/G contract, no
static-prefix/BOS tokens). It measures the same core quantities the paper reports
for generated trajectories:

  * Hit@1/5/10  -- one-step next-event prediction after the observed history
  * diagnosis Jaccard (set overlap) -- generated vs actual future diagnoses
  * first-event time MAE (days)     -- |generated[0].age - actual[0].age|
  * event-count MAE, sequence edit distance, death Brier

Usage:
  python mimic_generation_eval.py \
    --ckpt runs/m3_rel/checkpoints/last.pt \
    --data-dir mimic-iv/multitype --split test --device cuda \
    --out-dir results/eval_m3_rel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

def _find_repo(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "src" / "semantic_delphi_ukb" / "car_rope_model.py").exists():
            return parent
    return start

REPO_DIR = _find_repo(Path(__file__).resolve())
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.track_g_generation import (  # noqa: E402
    GeneratedEvent,
    expected_calibration_error,
    js_divergence,
    normalized_edit_distance,
    sample_event_and_wait_with_diagnostics,
    set_metrics,
    trajectory_validity_metrics,
)
from semantic_delphi_ukb.track_g_models import checkpoint_state  # noqa: E402
from utils import get_p2i  # noqa: E402


CLINICAL_TYPES = {"diagnosis", "procedure", "death"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-patients", type=int, default=None)
    p.add_argument("--num-rollouts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--followup-years", type=float, default=10.0)
    p.add_argument("--baseline-fraction", type=float, default=0.65)
    p.add_argument("--min-history-events", type=int, default=8)
    p.add_argument("--min-future-events", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--death-logit-bias", type=float, default=0.0)
    p.add_argument("--minimum-wait-days", type=float, default=1.0)
    p.add_argument("--sampling-seed", type=int, default=20260819)
    return p


@dataclass(frozen=True)
class Event:
    token_id: int
    age_days: float


def load_split(data_dir: Path, split: str):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    return data, p2i, static


def load_event_types(data_dir: Path) -> list[str]:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    vocab_csv = data_dir / manifest["vocab_csv"]
    vocab_size = int(manifest["vocab_size"])
    event_types = ["special"] * vocab_size
    with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tid = int(row["token_id"])
            if 0 <= tid < vocab_size:
                event_types[tid] = row["event_type"].strip()
    return event_types


def patient_events(data, p2i, patient_index: int) -> list[Event]:
    start, length = p2i[int(patient_index)]
    rows = np.asarray(data[int(start):int(start) + int(length)])
    order = np.argsort(rows[:, 1], kind="stable")
    return [Event(int(r[2]) + 1, float(r[1])) for r in rows[order]]


def choose_cases(data, p2i, args) -> list[tuple[int, int]]:
    cases = []
    for patient_index in range(len(p2i)):
        events = patient_events(data, p2i, patient_index)
        if len(events) < args.min_history_events + args.min_future_events:
            continue
        cut = int(math.floor((len(events) - 1) * args.baseline_fraction))
        cut = max(args.min_history_events - 1, cut)
        cut = min(cut, len(events) - args.min_future_events - 1)
        baseline_age = events[cut].age_days
        future = [e for e in events[cut + 1:] if e.age_days <= baseline_age + args.followup_years * 365.25]
        if len(future) < args.min_future_events:
            continue
        cases.append((patient_index, cut))
        if args.max_patients is not None and len(cases) >= args.max_patients:
            break
    return cases


def context_tensors(events: Sequence[Event], cut: int, block_size: int, device):
    history = events[: cut + 1][-block_size:]
    tokens = torch.tensor([[e.token_id for e in history]], dtype=torch.long, device=device)
    ages = torch.tensor([[e.age_days for e in history]], dtype=torch.float32, device=device)
    return tokens, ages


@torch.no_grad()
def last_logits(model, tokens, ages, static_features):
    logits, *_ = model(tokens, ages, static_features, None, None)
    return logits[0, -1]


@torch.no_grad()
def last_logits_batch(model, tokens, ages, static_features):
    logits, *_ = model(tokens, ages, static_features, None, None)
    return logits[:, -1]


def topk_metrics(logits, target, candidate_mask) -> dict:
    logits = logits[..., : candidate_mask.numel()]
    if not 0 <= int(target) < candidate_mask.numel():
        raise ValueError(f"target token {target} outside dynamic vocab")
    scores = logits.masked_fill(~candidate_mask, -torch.inf)
    order = torch.argsort(scores, descending=True)
    rank_pos = torch.where(order == int(target))[0]
    rank = int(rank_pos[0].item()) + 1 if rank_pos.numel() else int(scores.numel()) + 1
    return {
        "hit_at_1": int(rank <= 1),
        "hit_at_5": int(rank <= 5),
        "hit_at_10": int(rank <= 10),
        "reciprocal_rank": 1.0 / rank,
    }


@torch.no_grad()
def rollout_many(model, event_types, tokens, ages, static_features, candidate_mask, args, patient_index):
    count = int(args.num_rollouts)
    device = tokens.device
    death_mask = torch.tensor([et == "death" for et in event_types], dtype=torch.bool, device=device)
    generator = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    generator.manual_seed(int(args.sampling_seed) + 1_000_003 * patient_index)
    uniforms = torch.rand((count, int(args.max_new_tokens), 2), generator=generator, device=device)
    max_age = float(ages[0, -1].item()) + float(args.followup_years) * 365.25
    tokens = tokens.repeat(count, 1)
    ages = ages.repeat(count, 1)
    static_batch = static_features.repeat(count, 1)
    active = torch.ones(count, dtype=torch.bool, device=device)
    generated: list[list[GeneratedEvent]] = [[] for _ in range(count)]
    vocab_size = int(candidate_mask.numel())

    for step in range(int(args.max_new_tokens)):
        active_ids = torch.where(active)[0]
        if active_ids.numel() == 0:
            break
        logits = last_logits_batch(model, tokens[active_ids], ages[active_ids], static_batch[active_ids])
        logits = logits[..., :vocab_size]
        append_tokens = torch.zeros((count, 1), dtype=torch.long, device=device)
        append_ages = torch.full((count, 1), -10000.0, dtype=ages.dtype, device=device)
        for local_index, rollout_id_tensor in enumerate(active_ids):
            rollout_id = int(rollout_id_tensor.item())
            token_id, wait_days, wait_was_clamped = sample_event_and_wait_with_diagnostics(
                logits[local_index],
                candidate_mask=candidate_mask,
                ignore_tokens=model.config.ignore_tokens,
                t_min=model.config.t_min,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                death_token_mask=death_mask,
                death_logit_bias=float(args.death_logit_bias),
                minimum_wait_days=float(args.minimum_wait_days),
                event_uniform=uniforms[rollout_id, step, 0],
                wait_uniform=uniforms[rollout_id, step, 1],
            )
            next_age = float(ages[rollout_id, -1].item()) + wait_days
            if next_age > max_age:
                active[rollout_id] = False
                continue
            event_type = event_types[token_id]
            generated[rollout_id].append(GeneratedEvent(token_id, next_age, event_type, wait_was_clamped))
            append_tokens[rollout_id, 0] = token_id
            append_ages[rollout_id, 0] = next_age
            if event_type == "death":
                active[rollout_id] = False
        tokens = torch.cat((tokens, append_tokens), dim=1)
        ages = torch.cat((ages, append_ages), dim=1)
        if tokens.size(1) > model.config.block_size:
            tokens = tokens[:, -model.config.block_size:]
            ages = ages[:, -model.config.block_size:]
    return generated


def average(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def evaluate_case(model, event_types, events, case, static_features, candidate_mask, args, device):
    patient_index, cut = case
    history_end = events[cut].age_days
    future_limit = history_end + args.followup_years * 365.25
    actual = [e for e in events[cut + 1:] if e.age_days <= future_limit]
    actual_diag = [e for e in actual if event_types[e.token_id] == "diagnosis"]

    tokens, ages = context_tensors(events, cut, model.config.block_size, device)
    logits = last_logits(model, tokens, ages, static_features)
    logits = logits[..., : candidate_mask.numel()]

    if actual:
        one_step = topk_metrics(logits, actual[0].token_id, candidate_mask)
    else:
        one_step = {"hit_at_1": None, "hit_at_5": None, "hit_at_10": None, "reciprocal_rank": None}

    rollouts = rollout_many(
        model, event_types, tokens, ages, static_features, candidate_mask, args, patient_index
    )

    jaccards, precisions, recalls, edits, time_errors, count_errors, type_js, dup_rates, validity_rows = [], [], [], [], [], [], [], [], []
    death_count = 0
    for generated in rollouts:
        gen_diag = [e for e in generated if e.event_type == "diagnosis"]
        precision, recall, jaccard = set_metrics(
            {e.token_id for e in actual_diag}, {e.token_id for e in gen_diag}
        )
        jaccards.append(jaccard)
        precisions.append(precision)
        recalls.append(recall)
        edits.append(normalized_edit_distance(
            [e.token_id for e in actual], [e.token_id for e in generated]
        ))
        if generated and actual:
            time_errors.append(abs(generated[0].age_days - actual[0].age_days))
        count_errors.append(abs(len(generated) - len(actual)))
        type_js.append(js_divergence(
            [event_types[e.token_id] for e in actual], [event_types[e.token_id] for e in generated]
        ))
        dup_rates.append((len(generated) - len({e.token_id for e in generated})) / max(len(generated), 1))
        validity_rows.append(trajectory_validity_metrics(generated))
        death_count += int(any(e.event_type == "death" for e in generated))

    actual_death = int(any(event_types[e.token_id] == "death" for e in actual))
    death_probability = death_count / max(1, len(rollouts))
    return {
        "patient_index": int(patient_index),
        **one_step,
        "diagnosis_precision": average(precisions),
        "diagnosis_recall": average(recalls),
        "diagnosis_jaccard": average(jaccards),
        "sequence_edit_distance": average(edits),
        "first_event_time_mae_days": average(time_errors),
        "event_count_mae": average(count_errors),
        "event_type_js_divergence": average(type_js),
        "duplicate_event_rate": average(dup_rates),
        "nonmonotonic_time_rate": average([r["nonmonotonic_time_rate"] for r in validity_rows]),
        "post_death_event_rate": average([r["post_death_event_rate"] for r in validity_rows]),
        "minimum_wait_clamp_rate": average([r["minimum_wait_clamp_rate"] for r in validity_rows]),
        "death_probability": death_probability,
        "actual_death": actual_death,
        "death_brier": (death_probability - actual_death) ** 2,
    }


SCALAR_KEYS = [
    "hit_at_1", "hit_at_5", "hit_at_10", "reciprocal_rank",
    "diagnosis_precision", "diagnosis_recall", "diagnosis_jaccard",
    "sequence_edit_distance", "first_event_time_mae_days",
    "event_count_mae", "event_type_js_divergence", "duplicate_event_rate",
    "nonmonotonic_time_rate", "post_death_event_rate", "minimum_wait_clamp_rate",
    "death_brier",
]


def summarize(rows: list[dict]) -> dict:
    summary = {k: average([r[k] for r in rows]) for k in SCALAR_KEYS}
    summary["patient_count"] = len(rows)
    death_rows = [r for r in rows if r["death_probability"] is not None]
    summary["death_ece"] = (
        expected_calibration_error(
            [r["death_probability"] for r in death_rows],
            [r["actual_death"] for r in death_rows],
        ) if death_rows else None
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model, family = checkpoint_state(checkpoint, expected_family="carope")
    model = model.to(args.device).eval()

    data, p2i, static = load_split(args.data_dir, args.split)
    event_types = load_event_types(args.data_dir)
    candidate_mask = torch.tensor(
        [et in CLINICAL_TYPES for et in event_types], dtype=torch.bool, device=args.device
    )
    cases = choose_cases(data, p2i, args)

    rows = []
    for patient_index, cut in cases:
        events = patient_events(data, p2i, patient_index)
        static_features = torch.tensor(static[[patient_index]], dtype=torch.float32, device=args.device)
        rows.append(evaluate_case(
            model, event_types, events, (patient_index, cut),
            static_features, candidate_mask, args, args.device,
        ))

    summary = {
        "checkpoint": str(args.ckpt),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "family": family,
        "num_cases": len(cases),
        "config": {k: str(v) for k, v in vars(args).items()},
        "metrics": summarize(rows),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out_dir / "patient_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
