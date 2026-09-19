"""Generation evaluation for MIMIC Track-R CARoPE checkpoints (static prefix + BOS + additive_v2_2).

Same metrics as the simple-format evaluator, but the context prepends the static prefix
and a required dynamic BOS token, matching the Track-R TokenStatic contract.
"""
from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT

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

ROOT = str(REPO_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")

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
from semantic_delphi_ukb.track_r_batch import load_track_r_assets, load_track_r_bos_token_id  # noqa: E402
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
    p.add_argument("--eid-file", type=Path, default=None,
                   help="Restrict evaluation to the EIDs listed in this file (one per line), "
                        "enabling matched-patient comparisons across representations.")
    return p


@dataclass(frozen=True)
class Event:
    token_id: int
    age_days: float


def load_event_types(data_dir: Path) -> list[str]:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    vocab_size = int(manifest["vocab_size"])
    labels_path = data_dir / "labels.csv"
    if labels_path.exists():
        labels = [ln.rstrip("\n") for ln in labels_path.read_text(encoding="utf-8").splitlines()]
    else:
        labels = ["special"] * vocab_size
    event_types = ["special"] * vocab_size
    for i, lab in enumerate(labels):
        if i >= vocab_size:
            break
        if lab.startswith("phecode:"):
            event_types[i] = "diagnosis"
        elif lab.startswith("cpt:"):
            event_types[i] = "procedure"
        elif lab == "death:event":
            event_types[i] = "death"
        elif lab.startswith("composite:"):
            # Design C: visit-level composite token; classify by its first component code
            first = lab[len("composite:"):].split("|")[0]
            event_types[i] = "procedure" if first.startswith("cpt:") else "diagnosis"
        elif lab == "other:visit":
            # collapsed low-frequency visit composite -> visit-level clinical content
            event_types[i] = "diagnosis"
    return event_types


def load_split(data_dir: Path, split: str):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    prefix_ids, anchors = load_track_r_assets(data_dir, split)
    eids: list[str] = []
    index_path = data_dir / f"{split}_patient_index.csv"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                eids.append(str(row["eid"]))
    return data, p2i, prefix_ids, anchors, eids


def patient_events(data, p2i, patient_index: int) -> list[Event]:
    start, length = p2i[int(patient_index)]
    rows = np.asarray(data[int(start):int(start) + int(length)])
    order = np.argsort(rows[:, 1], kind="stable")
    return [Event(int(r[2]) + 1, float(r[1])) for r in rows[order]]


def choose_cases(data, p2i, args, eids=None, eid_set=None) -> list[tuple[int, int]]:
    cases = []
    for patient_index in range(len(p2i)):
        if eid_set is not None:
            if eids is None or patient_index >= len(eids) or eids[patient_index] not in eid_set:
                continue
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


def context_tensors(events, cut, prefix_ids, anchor_age, bos_id, dynamic_length, device):
    history = events[: cut + 1][-dynamic_length:]
    tokens = [int(v) for v in prefix_ids] + [int(bos_id)] + [e.token_id for e in history]
    ages = [float(anchor_age)] * len(prefix_ids) + [history[0].age_days] + [e.age_days for e in history]
    tt = torch.tensor([tokens], dtype=torch.long, device=device)
    at = torch.tensor([ages], dtype=torch.float32, device=device)
    static_mask = torch.zeros_like(tt, dtype=torch.bool)
    static_mask[:, : len(prefix_ids)] = True
    bos_mask = torch.zeros_like(tt, dtype=torch.bool)
    bos_mask[:, len(prefix_ids)] = True
    return tt, at, static_mask, bos_mask


@torch.no_grad()
def last_logits(model, tokens, ages, static_mask, bos_mask):
    logits, *_ = model(tokens, ages, None, static_token_mask=static_mask, bos_token_mask=bos_mask)
    return logits[0, -1]


@torch.no_grad()
def last_logits_batch(model, tokens, ages, static_mask, bos_mask):
    logits, *_ = model(tokens, ages, None, static_token_mask=static_mask, bos_token_mask=bos_mask)
    return logits[:, -1]


def topk_metrics(logits, target, candidate_mask) -> dict:
    logits = logits[..., : candidate_mask.numel()]
    scores = logits.masked_fill(~candidate_mask, -torch.inf)
    order = torch.argsort(scores, descending=True)
    rank_pos = torch.where(order == int(target))[0]
    rank = int(rank_pos[0].item()) + 1 if rank_pos.numel() else int(scores.numel()) + 1
    return {"hit_at_1": int(rank <= 1), "hit_at_5": int(rank <= 5), "hit_at_10": int(rank <= 10), "reciprocal_rank": 1.0 / rank}


@torch.no_grad()
def rollout_many(model, event_types, tokens, ages, static_mask, bos_mask, candidate_mask, args, patient_index):
    count = int(args.num_rollouts)
    device = tokens.device
    death_mask = torch.tensor([et == "death" for et in event_types], dtype=torch.bool, device=device)
    generator = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    generator.manual_seed(int(args.sampling_seed) + 1_000_003 * patient_index)
    uniforms = torch.rand((count, int(args.max_new_tokens), 2), generator=generator, device=device)
    max_age = float(ages[0, -1].item()) + float(args.followup_years) * 365.25
    tokens = tokens.repeat(count, 1)
    ages = ages.repeat(count, 1)
    static_mask = static_mask.repeat(count, 1)
    bos_mask = bos_mask.repeat(count, 1)
    active = torch.ones(count, dtype=torch.bool, device=device)
    generated: list[list[GeneratedEvent]] = [[] for _ in range(count)]
    vocab_size = int(candidate_mask.numel())
    static_count = int(static_mask[0].sum().item())

    for step in range(int(args.max_new_tokens)):
        active_ids = torch.where(active)[0]
        if active_ids.numel() == 0:
            break
        logits = last_logits_batch(model, tokens[active_ids], ages[active_ids], static_mask[active_ids], bos_mask[active_ids])
        logits = logits[..., :vocab_size]
        append_tokens = torch.zeros((count, 1), dtype=torch.long, device=device)
        append_ages = torch.full((count, 1), -10000.0, dtype=ages.dtype, device=device)
        for local_index, rollout_id_tensor in enumerate(active_ids):
            rollout_id = int(rollout_id_tensor.item())
            token_id, wait_days, wait_was_clamped = sample_event_and_wait_with_diagnostics(
                logits[local_index], candidate_mask=candidate_mask, ignore_tokens=model.config.ignore_tokens,
                t_min=model.config.t_min, temperature=float(args.temperature), top_p=float(args.top_p),
                death_token_mask=death_mask, death_logit_bias=float(args.death_logit_bias),
                minimum_wait_days=float(args.minimum_wait_days),
                event_uniform=uniforms[rollout_id, step, 0], wait_uniform=uniforms[rollout_id, step, 1])
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
        static_mask = torch.cat((static_mask, torch.zeros((count, 1), dtype=torch.bool, device=device)), dim=1)
        bos_mask = torch.cat((bos_mask, torch.zeros((count, 1), dtype=torch.bool, device=device)), dim=1)
        if tokens.size(1) > model.config.block_size:
            keep = model.config.block_size - static_count - 1
            prefix_t = tokens[:, : static_count + 1]
            prefix_a = ages[:, : static_count + 1]
            tokens = torch.cat((prefix_t, tokens[:, -keep:]), dim=1)
            ages = torch.cat((prefix_a, ages[:, -keep:]), dim=1)
            static_mask = torch.zeros_like(tokens, dtype=torch.bool); static_mask[:, :static_count] = True
            bos_mask = torch.zeros_like(tokens, dtype=torch.bool); bos_mask[:, static_count] = True
    return generated


def average(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def evaluate_case(model, event_types, events, case, prefix, anchor, bos_id, candidate_mask, args, device):
    patient_index, cut = case
    history_end = events[cut].age_days
    future_limit = history_end + args.followup_years * 365.25
    actual = [e for e in events[cut + 1:] if e.age_days <= future_limit]
    actual_diag = [e for e in actual if event_types[e.token_id] == "diagnosis"]
    tokens, ages, static_mask, bos_mask = context_tensors(
        events, cut, prefix, anchor, bos_id, 128, device)
    logits = last_logits(model, tokens, ages, static_mask, bos_mask)
    logits = logits[..., : candidate_mask.numel()]
    if actual:
        one_step = topk_metrics(logits, actual[0].token_id, candidate_mask)
    else:
        one_step = {"hit_at_1": None, "hit_at_5": None, "hit_at_10": None, "reciprocal_rank": None}

    rollouts = rollout_many(model, event_types, tokens, ages, static_mask, bos_mask, candidate_mask, args, patient_index)
    jaccards, precisions, recalls, edits, time_errors, count_errors, type_js, dup_rates, validity_rows = [], [], [], [], [], [], [], [], []
    death_count = 0
    for generated in rollouts:
        gen_diag = [e for e in generated if e.event_type == "diagnosis"]
        precision, recall, jaccard = set_metrics({e.token_id for e in actual_diag}, {e.token_id for e in gen_diag})
        jaccards.append(jaccard); precisions.append(precision); recalls.append(recall)
        edits.append(normalized_edit_distance([e.token_id for e in actual], [e.token_id for e in generated]))
        if generated and actual:
            time_errors.append(abs(generated[0].age_days - actual[0].age_days))
        count_errors.append(abs(len(generated) - len(actual)))
        type_js.append(js_divergence([event_types[e.token_id] for e in actual], [event_types[e.token_id] for e in generated]))
        dup_rates.append((len(generated) - len({e.token_id for e in generated})) / max(len(generated), 1))
        validity_rows.append(trajectory_validity_metrics(generated))
        death_count += int(any(e.event_type == "death" for e in generated))
    actual_death = int(any(event_types[e.token_id] == "death" for e in actual))
    death_probability = death_count / max(1, len(rollouts))
    return {
        "patient_index": int(patient_index), **one_step,
        "diagnosis_precision": average(precisions), "diagnosis_recall": average(recalls),
        "diagnosis_jaccard": average(jaccards), "sequence_edit_distance": average(edits),
        "first_event_time_mae_days": average(time_errors), "event_count_mae": average(count_errors),
        "event_type_js_divergence": average(type_js), "duplicate_event_rate": average(dup_rates),
        "nonmonotonic_time_rate": average([r["nonmonotonic_time_rate"] for r in validity_rows]),
        "post_death_event_rate": average([r["post_death_event_rate"] for r in validity_rows]),
        "minimum_wait_clamp_rate": average([r["minimum_wait_clamp_rate"] for r in validity_rows]),
        "death_probability": death_probability, "actual_death": actual_death,
        "death_brier": (death_probability - actual_death) ** 2,
    }


SCALAR_KEYS = ["hit_at_1", "hit_at_5", "hit_at_10", "reciprocal_rank", "diagnosis_precision", "diagnosis_recall",
               "diagnosis_jaccard", "sequence_edit_distance", "first_event_time_mae_days", "event_count_mae",
               "event_type_js_divergence", "duplicate_event_rate", "nonmonotonic_time_rate", "post_death_event_rate",
               "minimum_wait_clamp_rate", "death_brier"]


def summarize(rows):
    summary = {k: average([r[k] for r in rows]) for k in SCALAR_KEYS}
    summary["patient_count"] = len(rows)
    death_rows = [r for r in rows if r["death_probability"] is not None]
    summary["death_ece"] = expected_calibration_error([r["death_probability"] for r in death_rows], [r["actual_death"] for r in death_rows]) if death_rows else None
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    # expected_family=None -> auto-detect from the checkpoint's track_g_family
    # ("carope" for CARoPE A0/A2, "ethos_matched"/"foresight_matched" for matched baselines)
    model, family = checkpoint_state(checkpoint, expected_family=None)
    model = model.to(args.device).eval()
    data, p2i, prefix_ids, anchors, eids = load_split(args.data_dir, args.split)
    event_types = load_event_types(args.data_dir)
    candidate_mask = torch.tensor([et in CLINICAL_TYPES for et in event_types], dtype=torch.bool, device=args.device)
    bos_id = load_track_r_bos_token_id(args.data_dir)
    eid_set = None
    if args.eid_file is not None:
        eid_set = {ln.strip() for ln in args.eid_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    cases = choose_cases(data, p2i, args, eids=eids, eid_set=eid_set)
    rows = []
    for patient_index, cut in cases:
        events = patient_events(data, p2i, patient_index)
        row = evaluate_case(model, event_types, events, (patient_index, cut),
                            prefix_ids[patient_index], anchors[patient_index], bos_id,
                            candidate_mask, args, args.device)
        if eids and patient_index < len(eids):
            row["eid"] = eids[patient_index]
        rows.append(row)
    summary = {"checkpoint": str(args.ckpt), "data_dir": str(args.data_dir), "split": args.split,
               "family": family, "num_cases": len(cases),
               "eid_file": (str(args.eid_file) if args.eid_file is not None else None),
               "metrics": summarize(rows)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out_dir / "patient_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
