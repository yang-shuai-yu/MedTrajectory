"""Train a frozen-representation discrete death-hazard head on validation only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_track_g_generation import (  # noqa: E402
    context_tensors,
    load_split,
    patient_events,
)
from semantic_delphi_ukb.horizon_control_metrics import summarize_scores  # noqa: E402
from semantic_delphi_ukb.selected_disease_demo import load_labels  # noqa: E402
from semantic_delphi_ukb.track_g_contract import (  # noqa: E402
    accepted_checkpoint_protocol_hashes,
    assert_split_allowed,
    assert_training_allowed,
    load_track_g_protocol,
    model_spec,
)
from semantic_delphi_ukb.track_g_models import checkpoint_state  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_bos_token_id,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.train_architecture_risk_heads import load_followup_end_ages  # noqa: E402
from training_monitor import RunMonitor  # noqa: E402


VERSION = "track_g_death_hazard_head_v2_followup"
FIT_SPLIT_VERSION = "track_g_death_hazard_head_v1"
HORIZONS_YEARS = (1.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/track_g_v1/TRACK_G_v1.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def outer_role(patient_index: int) -> str:
    digest = hashlib.sha256(f"track_g_death_calibration_v1:{patient_index}".encode()).digest()
    return "fit" if int.from_bytes(digest[:8], "big") % 2 == 0 else "eval"


def fit_role(patient_index: int) -> str:
    if outer_role(patient_index) == "eval":
        return "eval"
    digest = hashlib.sha256(f"{FIT_SPLIT_VERSION}:{patient_index}".encode()).digest()
    return "head_train" if int.from_bytes(digest[:8], "big") % 2 == 0 else "calibration_fit"


def death_target(events, cut: int, followup_end_age: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline_age = float(events[cut].age_days)
    horizons = np.asarray(HORIZONS_YEARS, dtype=np.float64) * 365.25
    future = [event for event in events[cut + 1 :] if event.age_days > baseline_age]
    death_durations = [event.age_days - baseline_age for event in future if event.event_type == "death"]
    death_duration = min(death_durations) if death_durations else math.inf
    observed_event_duration = max((event.age_days - baseline_age for event in future), default=0.0)
    observation_duration = max(
        observed_event_duration,
        float(followup_end_age) - baseline_age if followup_end_age is not None else 0.0,
    )

    hazard_event = np.zeros(len(horizons), dtype=np.float32)
    hazard_mask = np.zeros(len(horizons), dtype=np.float32)
    previous = 0.0
    for index, upper in enumerate(horizons):
        if death_duration <= upper:
            if death_duration > previous:
                hazard_event[index] = 1.0
                hazard_mask[index] = 1.0
            break
        if observation_duration >= upper:
            hazard_mask[index] = 1.0
        else:
            break
        previous = upper

    horizon_event = np.zeros(len(horizons), dtype=np.float32)
    horizon_mask = np.zeros(len(horizons), dtype=np.float32)
    for index, upper in enumerate(horizons):
        if death_duration <= upper:
            horizon_event[index] = 1.0
            horizon_mask[index] = 1.0
        elif observation_duration >= upper:
            horizon_mask[index] = 1.0
    return hazard_event, hazard_mask, horizon_event, horizon_mask


@torch.no_grad()
def extract_feature(model, events, cut, prefix, anchor, bos_id, dynamic_length, device) -> np.ndarray:
    tokens, ages, static_mask, bos_mask = context_tensors(
        events, cut, prefix, anchor, bos_id, dynamic_length, device
    )
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.transformer.norm_f.register_forward_hook(hook)
    try:
        model(tokens, ages, None, static_token_mask=static_mask, bos_token_mask=bos_mask)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one final-norm activation, found {len(captured)}")
    return captured[0][0, -1].float().cpu().numpy()


class DeathHazardHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, len(HORIZONS_YEARS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    @staticmethod
    def cumulative_risk(logits: torch.Tensor) -> torch.Tensor:
        survival = torch.cumprod(1.0 - torch.sigmoid(logits), dim=-1)
        return 1.0 - survival


def masked_nll(logits: torch.Tensor, event: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, event, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def fit_shared_platt(probabilities: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    valid = masks.astype(bool)
    score = np.clip(probabilities[valid], 1.0e-5, 1.0 - 1.0e-5)
    target = labels[valid].astype(np.float64)
    logit = np.log(score) - np.log1p(-score)
    design = np.column_stack((np.ones(len(logit)), logit))
    beta = np.asarray((0.0, 1.0), dtype=np.float64)
    ridge = np.diag((1.0e-3, 1.0e-3))
    for _ in range(100):
        linear = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.maximum(probability * (1.0 - probability), 1.0e-8)
        working = design @ beta + (target - probability) / weight
        updated = np.linalg.solve(design.T @ (weight[:, None] * design) + ridge, design.T @ (weight * working))
        updated[0] = np.clip(updated[0], -10.0, 10.0)
        updated[1] = np.clip(updated[1], 0.05, 10.0)
        if np.max(np.abs(updated - beta)) < 1.0e-8:
            beta = updated
            break
        beta = updated
    return float(beta[0]), float(beta[1])


def apply_platt(probabilities: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1.0e-5, 1.0 - 1.0e-5)
    logit = np.log(clipped) - np.log1p(-clipped)
    linear = np.clip(intercept + slope * logit, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def classification_at_half(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= 0.5
    positive = labels.astype(bool)
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    return {
        "threshold": 0.5,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def metric_block(probabilities: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> dict:
    output = {}
    for index, horizon in enumerate(HORIZONS_YEARS):
        valid = masks[:, index].astype(bool)
        score = probabilities[valid, index]
        target = labels[valid, index].astype(np.int8)
        metrics = summarize_scores(score, target, bins=10, probability_scores=True)
        clipped = np.clip(score, 1.0e-7, 1.0 - 1.0e-7)
        metrics["log_loss"] = float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)))
        metrics["prevalence"] = float(target.mean())
        metrics.update(classification_at_half(score, target))
        output[f"{horizon:g}y"] = metrics
    output["monotonic_violation_count"] = int((np.diff(probabilities, axis=1) < -1.0e-8).sum())
    return output


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def self_test() -> int:
    events = [
        type("Event", (), {"age_days": 0.0, "event_type": "diagnosis"})(),
        type("Event", (), {"age_days": 2.0 * 365.25, "event_type": "death"})(),
    ]
    hazard_event, hazard_mask, horizon_event, horizon_mask = death_target(events, 0, 10.0 * 365.25)
    assert hazard_event.tolist() == [0.0, 1.0, 0.0]
    assert hazard_mask.tolist() == [1.0, 1.0, 0.0]
    assert horizon_event.tolist() == [0.0, 1.0, 1.0]
    assert horizon_mask.tolist() == [1.0, 1.0, 1.0]
    risk = DeathHazardHead.cumulative_risk(torch.tensor([[0.0, 1.0, -1.0]]))
    assert bool(torch.all(risk[:, 1:] >= risk[:, :-1]))
    print(json.dumps({"self_test": True, "monotonic": True}))
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    protocol = load_track_g_protocol(args.protocol, ROOT)
    assert_training_allowed(protocol)
    assert_split_allowed(protocol, "val")
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"unregistered seed: {args.seed}")
    spec = model_spec(protocol, args.model)
    checkpoint_path = ROOT / spec["checkpoint"].format(seed=args.seed)
    cohort_path = ROOT / protocol["cohort_manifests"]["validation"]
    cohort = json.loads(cohort_path.read_text(encoding="utf-8-sig"))
    if cohort.get("protocol_manifest_sha256") not in accepted_checkpoint_protocol_hashes(protocol):
        raise ValueError("validation cohort protocol hash mismatch")

    source = protocol["_source_track_r"]
    data_dir = Path(source["output_data_dir"])
    validate_track_r_data_manifest(data_dir, source)
    labels = load_labels(data_dir / "labels.csv")
    data, p2i, (prefix_ids, anchors) = load_split(data_dir, "val")
    observed_last_ages = np.asarray([
        float(np.asarray(data[int(start): int(start) + int(length), 1]).max())
        for start, length in p2i
    ], dtype=np.float32)
    followup_end_ages = load_followup_end_ages(data_dir, "val", observed_last_ages, 0)
    bos_id = load_track_r_bos_token_id(data_dir)
    cases = [(int(row["patient_index"]), int(row["cut_index"])) for row in cohort["cases"]]
    if args.max_patients > 0:
        cases = cases[: args.max_patients]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = {
        "version": VERSION,
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "cohort_protocol_manifest_sha256": cohort["protocol_manifest_sha256"],
        "horizons_years": list(HORIZONS_YEARS),
        "outer_split": "track_g_death_calibration_v1 hash parity: fit/eval",
        "fit_split": f"{FIT_SPLIT_VERSION} hash parity: head_train/calibration_fit",
        "censoring_source": str(data_dir / "val_followup_end_age_days.npy"),
        "max_epochs": args.max_epochs,
        "eval_interval": args.eval_interval,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "threshold": 0.5,
        "max_patients": args.max_patients,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    started = time.perf_counter()
    try:
        monitor.mark_running(phase="feature_extraction", model=args.model, seed=args.seed, extracted=0, total=len(cases))
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        expected_family = spec["family"] if spec["family"] != "carope" else "carope"
        model, _ = checkpoint_state(checkpoint, expected_family=expected_family)
        model = model.to(args.device).eval()
        features = []
        hazard_events = []
        hazard_masks = []
        horizon_events = []
        horizon_masks = []
        patient_indices = []
        roles = []
        dynamic_length = int(protocol["matched_input_contract"]["dynamic_context_length"])
        for index, (patient_index, cut) in enumerate(cases, start=1):
            events = patient_events(data, p2i, patient_index, labels)
            features.append(extract_feature(
                model, events, cut, prefix_ids[patient_index], anchors[patient_index], bos_id,
                dynamic_length, args.device,
            ))
            target = death_target(events, cut, float(followup_end_ages[patient_index]))
            hazard_events.append(target[0]); hazard_masks.append(target[1])
            horizon_events.append(target[2]); horizon_masks.append(target[3])
            patient_indices.append(patient_index); roles.append(fit_role(patient_index))
            if index % 100 == 0 or index == len(cases):
                monitor.update_status(status="running", phase="feature_extraction", extracted=index, total=len(cases))
        del model, checkpoint
        torch.cuda.empty_cache()

        feature_array = np.asarray(features, dtype=np.float32)
        hazard_event_array = np.asarray(hazard_events, dtype=np.float32)
        hazard_mask_array = np.asarray(hazard_masks, dtype=np.float32)
        horizon_event_array = np.asarray(horizon_events, dtype=np.float32)
        horizon_mask_array = np.asarray(horizon_masks, dtype=np.float32)
        role_array = np.asarray(roles)
        train_mask = role_array == "head_train"
        calibration_mask = role_array == "calibration_fit"
        eval_mask = role_array == "eval"
        if min(train_mask.sum(), calibration_mask.sum(), eval_mask.sum()) == 0:
            raise RuntimeError("one or more validation roles are empty")
        mean = feature_array[train_mask].mean(axis=0)
        scale = feature_array[train_mask].std(axis=0)
        scale[scale < 1.0e-6] = 1.0
        standardized = (feature_array - mean) / scale

        x = torch.tensor(standardized, dtype=torch.float32, device=args.device)
        hazard_y = torch.tensor(hazard_event_array, dtype=torch.float32, device=args.device)
        hazard_m = torch.tensor(hazard_mask_array, dtype=torch.float32, device=args.device)
        train_index = torch.tensor(train_mask, dtype=torch.bool, device=args.device)
        calibration_index = torch.tensor(calibration_mask, dtype=torch.bool, device=args.device)
        head = DeathHazardHead(x.size(1)).to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        monitor.mark_running(
            phase="head_training", model=args.model, seed=args.seed, epoch=0,
            head_train_count=int(train_mask.sum()), calibration_fit_count=int(calibration_mask.sum()),
            eval_count=int(eval_mask.sum()),
        )
        for epoch in range(1, args.max_epochs + 1):
            head.train()
            logits = head(x[train_index])
            loss = masked_nll(logits, hazard_y[train_index], hazard_m[train_index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            monitor.log_step(epoch, {
                "loss/train_total": loss,
                "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                "optimization/gradient_norm": gradient_norm,
            })
            if epoch % args.eval_interval != 0 and epoch != args.max_epochs:
                continue
            head.eval()
            with torch.no_grad():
                calibration_loss = masked_nll(
                    head(x[calibration_index]),
                    hazard_y[calibration_index],
                    hazard_m[calibration_index],
                )
            row = {
                "epoch": epoch,
                "global_step": epoch,
                "train_loss": float(loss.detach().cpu()),
                "calibration_fit_nll": float(calibration_loss.cpu()),
                "best_calibration_fit_nll": min(best_loss, float(calibration_loss.cpu())),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            monitor.log_epoch(row)
            state = {
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_calibration_fit_nll": best_loss,
                "feature_mean": mean,
                "feature_scale": scale,
                "config": config,
                "random_state": monitor.capture_random_state(),
            }
            monitor.save_checkpoint(state, "last.pt")
            value = float(calibration_loss.cpu())
            if value < best_loss - 1.0e-7:
                best_loss = value
                best_epoch = epoch
                stale = 0
                state.update(best_epoch=best_epoch, best_calibration_fit_nll=best_loss)
                monitor.save_checkpoint(state, "best_calibration_fit_nll.pt")
            else:
                stale += args.eval_interval
            monitor.update_status(
                status="running", phase="head_training", epoch=epoch, best_epoch=best_epoch,
                calibration_fit_nll=value, best_calibration_fit_nll=best_loss, patience_used=stale,
            )
            if stale >= args.patience:
                break

        best = torch.load(
            args.run_dir / "checkpoints/best_calibration_fit_nll.pt",
            map_location=args.device,
            weights_only=False,
        )
        head.load_state_dict(best["head"])
        head.eval()
        with torch.no_grad():
            raw_probability = DeathHazardHead.cumulative_risk(head(x)).cpu().numpy()
        intercept, slope = fit_shared_platt(
            raw_probability[calibration_mask],
            horizon_event_array[calibration_mask],
            horizon_mask_array[calibration_mask],
        )
        calibrated_probability = apply_platt(raw_probability, intercept, slope)
        summary = {
            "version": VERSION,
            "model": args.model,
            "seed": args.seed,
            "split_counts": {role: int((role_array == role).sum()) for role in ("head_train", "calibration_fit", "eval")},
            "best_epoch": best_epoch,
            "best_calibration_fit_nll": best_loss,
            "platt": {"shared_across_horizons": True, "intercept": intercept, "slope": slope},
            "raw_eval": metric_block(raw_probability[eval_mask], horizon_event_array[eval_mask], horizon_mask_array[eval_mask]),
            "calibrated_eval": metric_block(calibrated_probability[eval_mask], horizon_event_array[eval_mask], horizon_mask_array[eval_mask]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.run_dir / "summary.json", summary)
        with gzip.open(args.run_dir / "eval_rows.json.gz", "wt", encoding="utf-8", newline="\n") as handle:
            for index in np.flatnonzero(eval_mask):
                row = {
                    "patient_index": int(patient_indices[index]),
                    "labels": horizon_event_array[index].astype(int).tolist(),
                    "masks": horizon_mask_array[index].astype(int).tolist(),
                    "raw_probability": raw_probability[index].tolist(),
                    "calibrated_probability": calibrated_probability[index].tolist(),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        best["platt"] = {"intercept": intercept, "slope": slope}
        monitor.save_checkpoint(best, "best_calibrated.pt")
        monitor.mark_finished(
            phase="finished", model=args.model, seed=args.seed, epoch=best_epoch,
            best_calibration_fit_nll=best_loss, elapsed_seconds=summary["elapsed_seconds"],
        )
        print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    except BaseException as exc:
        monitor.mark_failed(exc, model=args.model, seed=args.seed)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
