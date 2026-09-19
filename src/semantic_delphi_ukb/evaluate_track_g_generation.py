"""Evaluate one Track G model/seed on validation or an explicitly authorized test split."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.selected_disease_demo import load_labels  # noqa: E402
from semantic_delphi_ukb.track_g_contract import (  # noqa: E402
    accepted_checkpoint_protocol_hashes,
    assert_split_allowed,
    assert_training_allowed,
    load_track_g_protocol,
    model_spec,
)
from semantic_delphi_ukb.track_g_generation import (  # noqa: E402
    GeneratedEvent,
    clinical_candidate_mask,
    clinical_event_type,
    expected_calibration_error,
    js_divergence,
    normalized_edit_distance,
    sample_event_and_wait_with_diagnostics,
    set_metrics,
    trajectory_validity_metrics,
    waiting_time_nll,
)
from semantic_delphi_ukb.track_g_models import checkpoint_state  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_assets,
    load_track_r_bos_token_id,
    validate_track_r_data_manifest,
)
from utils import get_p2i  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=REPO_DIR / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument("--model", required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--data-dir", type=Path, default=None)
    value.add_argument("--split", choices=("val", "test"), default="val")
    value.add_argument("--out-dir", type=Path, required=True)
    value.add_argument("--device", default="cpu")
    value.add_argument("--max-patients", type=int, default=None)
    value.add_argument("--num-rollouts", type=int, default=None)
    value.add_argument("--temperature", type=float, default=1.0)
    value.add_argument("--top-p", type=float, default=1.0)
    value.add_argument("--death-logit-bias", type=float, default=0.0)
    value.add_argument(
        "--exclude-death-token",
        action="store_true",
        help="Validation-only ablation: remove death from event identity and waiting-time rate.",
    )
    value.add_argument("--sampler-manifest", type=Path, default=None)
    value.add_argument("--cohort-manifest", type=Path, default=None)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_split(data_dir: Path, split: str):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    return data, get_p2i(data), load_track_r_assets(data_dir, split)


def patient_events(data: np.ndarray, p2i: np.ndarray, patient_index: int, labels: Sequence[str]) -> list[GeneratedEvent]:
    start, length = p2i[patient_index]
    rows = np.asarray(data[int(start) : int(start) + int(length)])
    order = np.argsort(rows[:, 1], kind="stable")
    events = []
    for row in rows[order]:
        token_id = int(row[2]) + 1
        label = labels[token_id] if token_id < len(labels) else str(token_id)
        event_type = clinical_event_type(label)
        if event_type != "other":
            events.append(GeneratedEvent(token_id, float(row[1]), event_type))
    return events


def eligible_cases(data, p2i, labels, settings: dict, max_patients: int, seed: int) -> list[tuple[int, int]]:
    eligible = []
    for patient_index in range(len(p2i)):
        events = patient_events(data, p2i, patient_index, labels)
        if len(events) < int(settings["min_history_events"]) + int(settings["min_future_events"]):
            continue
        cut = int(math.floor((len(events) - 1) * float(settings["baseline_fraction"])))
        cut = max(int(settings["min_history_events"]) - 1, cut)
        cut = min(cut, len(events) - int(settings["min_future_events"]) - 1)
        baseline_age = events[cut].age_days
        limit = baseline_age + float(settings["followup_years"]) * 365.25
        future = [event for event in events[cut + 1 :] if event.age_days <= limit]
        if len(future) >= int(settings["min_future_events"]):
            eligible.append((patient_index, cut))
    if max_patients > 0 and len(eligible) > max_patients:
        generator = np.random.default_rng(seed)
        selected = generator.choice(len(eligible), size=max_patients, replace=False)
        eligible = [eligible[int(index)] for index in selected]
    return eligible


def context_tensors(events, cut, prefix_ids, anchor_age, bos_token_id, dynamic_length, device):
    history = events[: cut + 1][-dynamic_length:]
    tokens = [int(value) for value in prefix_ids] + [int(bos_token_id)] + [event.token_id for event in history]
    ages = [float(anchor_age)] * len(prefix_ids) + [history[0].age_days] + [event.age_days for event in history]
    token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    age_tensor = torch.tensor([ages], dtype=torch.float32, device=device)
    static_mask = torch.zeros_like(token_tensor, dtype=torch.bool)
    static_mask[:, : len(prefix_ids)] = True
    bos_mask = torch.zeros_like(token_tensor, dtype=torch.bool)
    bos_mask[:, len(prefix_ids)] = True
    return token_tensor, age_tensor, static_mask, bos_mask


@torch.no_grad()
def last_logits(model, tokens, ages, static_mask, bos_mask):
    logits, *_ = model(
        tokens,
        ages,
        None,
        static_token_mask=static_mask,
        bos_token_mask=bos_mask,
    )
    return logits[0, -1]


@torch.no_grad()
def last_logits_batch(model, tokens, ages, static_mask, bos_mask):
    logits, *_ = model(
        tokens,
        ages,
        None,
        static_token_mask=static_mask,
        bos_token_mask=bos_mask,
    )
    return logits[:, -1]


def topk_metrics(logits, target, candidate_mask) -> dict:
    logits = dynamic_vocab_logits(logits, candidate_mask)
    if not 0 <= int(target) < candidate_mask.numel():
        raise ValueError(f"target token {target} is outside the dynamic vocabulary")
    scores = logits.masked_fill(~candidate_mask, -torch.inf)
    order = torch.argsort(scores, descending=True)
    rank_position = torch.where(order == int(target))[0]
    rank = int(rank_position[0].item()) + 1 if int(rank_position.numel()) else scores.numel() + 1
    return {
        "hit_at_1": int(rank <= 1),
        "hit_at_5": int(rank <= 5),
        "hit_at_10": int(rank <= 10),
        "reciprocal_rank": 1.0 / rank,
    }


def dynamic_vocab_logits(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    dynamic_vocab_size = int(candidate_mask.numel())
    if logits.size(-1) < dynamic_vocab_size:
        raise ValueError(
            f"model vocabulary {logits.size(-1)} is smaller than dynamic vocabulary {dynamic_vocab_size}"
        )
    return logits[..., :dynamic_vocab_size]


@torch.no_grad()
def rollout_many(
    model,
    labels,
    tokens,
    ages,
    static_mask,
    bos_mask,
    candidate_mask,
    settings,
    sampler,
    patient_index,
):
    count = int(sampler["num_rollouts"])
    generated = [[] for _ in range(count)]
    device = tokens.device
    death_mask = torch.tensor(
        [clinical_event_type(label) == "death" for label in labels], dtype=torch.bool, device=device
    )
    generator = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    generator.manual_seed(int(settings["sampling_seed"]) + 1_000_003 * patient_index)
    uniforms = torch.rand(
        (count, int(settings["max_new_tokens"]), 2), generator=generator, device=device
    )
    max_age = float(ages[0, -1].item()) + float(settings["followup_years"]) * 365.25
    tokens = tokens.repeat(count, 1)
    ages = ages.repeat(count, 1)
    static_mask = static_mask.repeat(count, 1)
    bos_mask = bos_mask.repeat(count, 1)
    active = torch.ones(count, dtype=torch.bool, device=device)
    for step in range(int(settings["max_new_tokens"])):
        active_ids = torch.where(active)[0]
        if int(active_ids.numel()) == 0:
            break
        logits = last_logits_batch(
            model,
            tokens[active_ids],
            ages[active_ids],
            static_mask[active_ids],
            bos_mask[active_ids],
        )
        logits = dynamic_vocab_logits(logits, candidate_mask)
        append_tokens = torch.zeros((count, 1), dtype=torch.long, device=device)
        append_ages = torch.full((count, 1), -10000.0, dtype=ages.dtype, device=device)
        for local_index, rollout_id_tensor in enumerate(active_ids):
            rollout_id = int(rollout_id_tensor.item())
            token_id, wait_days, wait_was_clamped = sample_event_and_wait_with_diagnostics(
                logits[local_index],
                candidate_mask=candidate_mask,
                ignore_tokens=model.config.ignore_tokens,
                t_min=model.config.t_min,
                temperature=float(sampler["temperature"]),
                top_p=float(sampler["top_p"]),
                death_token_mask=death_mask,
                death_logit_bias=float(sampler["death_logit_bias"]),
                minimum_wait_days=float(settings["minimum_wait_days"]),
                event_uniform=uniforms[rollout_id, step, 0],
                wait_uniform=uniforms[rollout_id, step, 1],
                rate_candidate_mask=candidate_mask if sampler.get("exclude_death_token", False) else None,
            )
            next_age = float(ages[rollout_id, -1].item()) + wait_days
            if next_age > max_age:
                active[rollout_id] = False
                continue
            event_type = clinical_event_type(labels[token_id])
            generated[rollout_id].append(
                GeneratedEvent(token_id, next_age, event_type, wait_was_clamped=wait_was_clamped)
            )
            append_tokens[rollout_id, 0] = token_id
            append_ages[rollout_id, 0] = next_age
            if event_type == "death" and settings["stop_after_death"]:
                active[rollout_id] = False
        tokens = torch.cat((tokens, append_tokens), dim=1)
        ages = torch.cat((ages, append_ages), dim=1)
        static_mask = torch.cat((static_mask, torch.zeros((count, 1), dtype=torch.bool, device=device)), dim=1)
        bos_mask = torch.cat((bos_mask, torch.zeros((count, 1), dtype=torch.bool, device=device)), dim=1)
        if tokens.size(1) > model.config.block_size:
            static_count = int(static_mask[0].sum().item())
            keep_dynamic = model.config.block_size - static_count - 1
            prefix_tokens = tokens[:, : static_count + 1]
            prefix_ages = ages[:, : static_count + 1]
            tokens = torch.cat((prefix_tokens, tokens[:, -keep_dynamic:]), dim=1)
            ages = torch.cat((prefix_ages, ages[:, -keep_dynamic:]), dim=1)
            static_mask = torch.zeros_like(tokens, dtype=torch.bool); static_mask[:, :static_count] = True
            bos_mask = torch.zeros_like(tokens, dtype=torch.bool); bos_mask[:, static_count] = True
    return generated


def average(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(finite) if finite else float("nan")


def one_step_waiting_time_metrics(logits, target_age, history_end, ignore_tokens, t_min) -> dict:
    delta_days = float(target_age) - float(history_end)
    if delta_days <= 0.0:
        return {"waiting_time_nll": None, "waiting_time_nll_valid": 0}
    value = waiting_time_nll(logits, delta_days, ignore_tokens, t_min)
    return {"waiting_time_nll": float(value.cpu()), "waiting_time_nll_valid": 1}


def evaluate_case(model, labels, events, case, prefix, anchor, bos_id, candidate_mask, settings, sampler, device):
    patient_index, cut = case
    history_end = events[cut].age_days
    future_limit = history_end + float(settings["followup_years"]) * 365.25
    actual = [event for event in events[cut + 1 :] if event.age_days <= future_limit]
    actual_trajectory = (
        [event for event in actual if event.event_type != "death"]
        if sampler.get("exclude_death_token", False)
        else actual
    )
    actual_diag = [event for event in actual_trajectory if event.event_type == "diagnosis"]
    tokens, ages, static_mask, bos_mask = context_tensors(
        events,
        cut,
        prefix,
        anchor,
        bos_id,
        int(settings["dynamic_context_length"]),
        device,
    )
    logits = dynamic_vocab_logits(
        last_logits(model, tokens, ages, static_mask, bos_mask), candidate_mask
    )
    if actual_trajectory:
        one_step = topk_metrics(logits, actual_trajectory[0].token_id, candidate_mask)
        rate_logits = (
            logits.masked_fill(~candidate_mask, -torch.inf)
            if sampler.get("exclude_death_token", False)
            else logits
        )
        one_step.update(
            one_step_waiting_time_metrics(
                rate_logits,
                actual_trajectory[0].age_days,
                history_end,
                model.config.ignore_tokens,
                model.config.t_min,
            )
        )
    else:
        one_step = {
            "hit_at_1": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "reciprocal_rank": None,
            "waiting_time_nll": None,
            "waiting_time_nll_valid": 0,
        }

    precisions = []; recalls = []; jaccards = []; edits = []; time_errors = []
    count_errors = []; type_js = []; duplicate_rates = []; validity_rows = []
    death_count = 0
    generated_rollouts = rollout_many(
        model,
        labels,
        tokens,
        ages,
        static_mask,
        bos_mask,
        candidate_mask,
        settings,
        sampler,
        patient_index,
    )
    for generated in generated_rollouts:
        generated_diag = [event for event in generated if event.event_type == "diagnosis"]
        precision, recall, jaccard = set_metrics(
            {event.token_id for event in actual_diag}, {event.token_id for event in generated_diag}
        )
        precisions.append(precision); recalls.append(recall); jaccards.append(jaccard)
        edits.append(normalized_edit_distance([event.token_id for event in actual_trajectory], [event.token_id for event in generated]))
        if generated and actual_trajectory:
            time_errors.append(abs(generated[0].age_days - actual_trajectory[0].age_days))
        count_errors.append(abs(len(generated) - len(actual_trajectory)))
        type_js.append(js_divergence([event.event_type for event in actual_trajectory], [event.event_type for event in generated]))
        duplicate_rates.append((len(generated) - len({event.token_id for event in generated})) / max(len(generated), 1))
        validity_rows.append(trajectory_validity_metrics(generated))
        death_count += int(any(event.event_type == "death" for event in generated))

    actual_death = int(any(event.event_type == "death" for event in actual))
    death_probability = death_count / int(sampler["num_rollouts"])
    return {
        "patient_index": int(patient_index),
        "target_event_type": actual_trajectory[0].event_type if actual_trajectory else "death_only",
        **one_step,
        "diagnosis_precision": average(precisions),
        "diagnosis_recall": average(recalls),
        "diagnosis_jaccard": average(jaccards),
        "sequence_edit_distance": average(edits),
        "first_event_time_mae_days": average(time_errors),
        "event_count_mae": average(count_errors),
        "event_type_js_divergence": average(type_js),
        "duplicate_event_rate": average(duplicate_rates),
        "nonmonotonic_time_rate": average([row["nonmonotonic_time_rate"] for row in validity_rows]),
        "post_death_event_rate": average([row["post_death_event_rate"] for row in validity_rows]),
        "minimum_wait_clamp_rate": average([row["minimum_wait_clamp_rate"] for row in validity_rows]),
        "death_probability": None if sampler.get("exclude_death_token", False) else death_probability,
        "actual_death": actual_death,
        "death_brier": None if sampler.get("exclude_death_token", False) else (death_probability - actual_death) ** 2,
    }


def summarize(rows: list[dict]) -> dict:
    scalar_keys = [
        "hit_at_1", "hit_at_5", "hit_at_10", "reciprocal_rank", "waiting_time_nll",
        "diagnosis_precision", "diagnosis_recall", "diagnosis_jaccard", "sequence_edit_distance",
        "first_event_time_mae_days", "event_count_mae", "event_type_js_divergence",
        "duplicate_event_rate", "nonmonotonic_time_rate", "post_death_event_rate",
        "minimum_wait_clamp_rate", "death_brier",
    ]
    summary = {key: average([row[key] for row in rows]) for key in scalar_keys}
    summary["patient_count"] = len(rows)
    summary["waiting_time_nll_valid_count"] = sum(row["waiting_time_nll_valid"] for row in rows)
    death_rows = [row for row in rows if row["death_probability"] is not None]
    if not death_rows:
        summary["death_brier"] = None
    summary["death_ece"] = (
        expected_calibration_error(
            [row["death_probability"] for row in death_rows],
            [row["actual_death"] for row in death_rows],
        )
        if death_rows
        else None
    )
    summary["trajectory_target_valid_count"] = sum(
        row["target_event_type"] != "death_only" for row in rows
    )
    summary["target_type_hit_at_k"] = {}
    for event_type in ("diagnosis", "procedure", "cancer", "death"):
        subset = [row for row in rows if row["target_event_type"] == event_type]
        summary["target_type_hit_at_k"][event_type] = {
            "n": len(subset),
            **{key: average([row[key] for row in subset]) for key in ("hit_at_1", "hit_at_5", "hit_at_10")},
        }
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.exclude_death_token and args.split != "val":
        raise ValueError("--exclude-death-token is restricted to validation")
    protocol = load_track_g_protocol(args.protocol, REPO_DIR)
    assert_training_allowed(protocol)
    assert_split_allowed(protocol, args.split)
    spec = model_spec(protocol, args.model)
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"unregistered Track G seed: {args.seed}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    expected_family = spec["family"] if spec["family"] != "carope" else "carope"
    model, _ = checkpoint_state(checkpoint, expected_family=expected_family)
    model = model.to(args.device).eval()

    source = protocol["_source_track_r"]
    data_dir = args.data_dir or Path(source["output_data_dir"])
    validate_track_r_data_manifest(data_dir, source)
    labels = load_labels(data_dir / "labels.csv")
    data, p2i, (prefix_ids, anchors) = load_split(data_dir, args.split)
    bos_id = load_track_r_bos_token_id(data_dir)
    settings = dict(protocol["generation_evaluation"])
    settings["dynamic_context_length"] = int(protocol["matched_input_contract"]["dynamic_context_length"])
    evaluation = settings["validation" if args.split == "val" else "locked_test"]
    max_patients = int(args.max_patients if args.max_patients is not None else evaluation["max_patients"])
    num_rollouts = int(args.num_rollouts if args.num_rollouts is not None else evaluation["num_rollouts"])
    if args.cohort_manifest is not None:
        cohort = json.loads(args.cohort_manifest.read_text(encoding="utf-8-sig"))
        if cohort.get("protocol_manifest_sha256") not in accepted_checkpoint_protocol_hashes(protocol):
            raise ValueError("cohort manifest protocol hash mismatch")
        if cohort.get("split") != args.split:
            raise ValueError("cohort manifest split mismatch")
        cases = [(int(row["patient_index"]), int(row["cut_index"])) for row in cohort["cases"]]
        if max_patients > 0:
            cases = cases[:max_patients]
    else:
        cases = eligible_cases(data, p2i, labels, settings, max_patients, int(settings["sampling_seed"]))
    candidate_mask = clinical_candidate_mask(labels, args.device)
    if args.exclude_death_token:
        death_mask = torch.tensor(
            [clinical_event_type(label) == "death" for label in labels],
            dtype=torch.bool,
            device=args.device,
        )
        candidate_mask = candidate_mask & ~death_mask
    sampler = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "death_logit_bias": args.death_logit_bias,
        "num_rollouts": num_rollouts,
        "exclude_death_token": bool(args.exclude_death_token),
    }
    if args.sampler_manifest is not None:
        manifest = json.loads(args.sampler_manifest.read_text(encoding="utf-8-sig"))
        accepted_hashes = accepted_checkpoint_protocol_hashes(protocol)
        if manifest.get("protocol_manifest_sha256") not in accepted_hashes:
            raise ValueError("sampler manifest protocol hash mismatch")
        selected = manifest["selected_by_family"][spec["family"]]
        sampler.update({key: selected[key] for key in ("temperature", "top_p", "death_logit_bias")})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "status.json", {"status": "running", "split": args.split, "model": args.model, "seed": args.seed})
    rows = []
    try:
        for case in cases:
            patient_index, _ = case
            events = patient_events(data, p2i, patient_index, labels)
            rows.append(
                evaluate_case(
                    model,
                    labels,
                    events,
                    case,
                    prefix_ids[patient_index],
                    anchors[patient_index],
                    bos_id,
                    candidate_mask,
                    settings,
                    sampler,
                    args.device,
                )
            )
        summary = {
            "protocol_id": protocol["protocol_id"],
            "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
            "split": args.split,
            "model": args.model,
            "family": spec["family"],
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "sampler": sampler,
            "cohort_protocol_manifest_sha256": (
                cohort.get("protocol_manifest_sha256") if args.cohort_manifest is not None else None
            ),
            "sampler_protocol_manifest_sha256": (
                manifest.get("protocol_manifest_sha256") if args.sampler_manifest is not None else None
            ),
            "metrics": summarize(rows),
        }
        write_rows(args.out_dir / "patient_rows.json.gz", rows)
        atomic_json(args.out_dir / "summary.json", summary)
        atomic_json(args.out_dir / "status.json", {"status": "finished", "patient_count": len(rows)})
    except BaseException as exc:
        atomic_json(args.out_dir / "status.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
